"""
Database Query Optimization and Indexing for Multi-Agent Systems.

Provides optimized queries, indexes, and caching strategies for high-performance
multi-agent coordination scenarios.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.enhanced_connection_manager import get_enhanced_db_manager
from database.models import (
    Agent,
    AgentStatus,
    Coalition,
    CoalitionStatus,
    agent_coalition_association,
)

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Optimized database queries for multi-agent coordination."""

    def __init__(self):
        """Initialize query optimizer."""
        self.db_manager = get_enhanced_db_manager()
        self.query_cache = {}
        self.query_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_query_time": 0.0,
            "slow_queries": [],
        }

    async def create_performance_indexes(self, session: AsyncSession):
        """Create performance indexes for multi-agent queries."""
        indexes = [
            # Agent performance indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_status_created ON agents (status, created_at)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_last_active ON agents (last_active) WHERE last_active IS NOT NULL",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_template_status ON agents (template, status)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_name_trgm ON agents USING gin (name gin_trgm_ops)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_inference_count ON agents (inference_count DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_total_steps ON agents (total_steps DESC)",
            # Coalition performance indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_status_created ON coalitions (status, created_at)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_performance_score ON coalitions (performance_score DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_cohesion_score ON coalitions (cohesion_score DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_name_trgm ON coalitions USING gin (name gin_trgm_ops)",
            # Agent-Coalition association indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_agent_id ON agent_coalition (agent_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_coalition_id ON agent_coalition (coalition_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_role ON agent_coalition (role)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_joined_at ON agent_coalition (joined_at)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_contribution ON agent_coalition (contribution_score DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_trust ON agent_coalition (trust_score DESC)",
            # Knowledge graph indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_type ON db_knowledge_nodes (type)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_creator ON db_knowledge_nodes (creator_agent_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_created ON db_knowledge_nodes (created_at)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_updated ON db_knowledge_nodes (updated_at)",
            # Composite indexes for common query patterns
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_status_active_updated ON agents (status, last_active, updated_at) WHERE status = 'active'",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_active_performance ON coalitions (status, performance_score, created_at) WHERE status = 'active'",
            # Partial indexes for common filters
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_active_only ON agents (id, name, created_at) WHERE status = 'active'",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_active_only ON coalitions (id, name, created_at) WHERE status = 'active'",
            # JSON field indexes (PostgreSQL-specific)
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_beliefs_gin ON agents USING gin (beliefs)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_preferences_gin ON agents USING gin (preferences)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_metrics_gin ON agents USING gin (metrics)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_objectives_gin ON coalitions USING gin (objectives)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_capabilities_gin ON coalitions USING gin (required_capabilities)",
        ]

        # Enable PostgreSQL extensions if needed
        extensions = [
            "CREATE EXTENSION IF NOT EXISTS pg_trgm",  # For text search
            "CREATE EXTENSION IF NOT EXISTS btree_gin",  # For GIN indexes
        ]

        # Create extensions first
        for ext in extensions:
            try:
                await session.execute(text(ext))
                await session.commit()
                logger.info(f"Created extension: {ext}")
            except Exception as e:
                logger.warning(f"Extension creation failed (may already exist): {e}")
                await session.rollback()

        # Create indexes
        for index_sql in indexes:
            try:
                await session.execute(text(index_sql))
                await session.commit()
                logger.info(f"Created index: {index_sql}")
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")
                await session.rollback()

    async def get_active_agents_optimized(
        self,
        session: AsyncSession,
        limit: int = 100,
        offset: int = 0,
        include_metrics: bool = True,
    ) -> List[Agent]:
        """Get active agents with optimized query."""
        start_time = time.time()

        # Use partial index for active agents
        query = select(Agent).where(Agent.status == AgentStatus.ACTIVE)

        if include_metrics:
            # Order by activity for better performance
            query = query.order_by(desc(Agent.last_active), desc(Agent.created_at))
        else:
            # If metrics not needed, use simpler ordering
            query = query.order_by(desc(Agent.created_at))

        query = query.limit(limit).offset(offset)

        result = await session.execute(query)
        agents = result.scalars().all()

        # Track query performance
        query_time = time.time() - start_time
        self._track_query_performance("get_active_agents", query_time)

        return list(agents)

    async def get_agent_coalitions_optimized(
        self,
        session: AsyncSession,
        agent_id: str,
        include_inactive: bool = False,
    ) -> List[Coalition]:
        """Get agent's coalitions with optimized joins."""
        start_time = time.time()

        # Use optimized join with association table
        query = (
            select(Coalition)
            .join(
                agent_coalition_association,
                Coalition.id == agent_coalition_association.c.coalition_id,
            )
            .where(agent_coalition_association.c.agent_id == agent_id)
        )

        if not include_inactive:
            query = query.where(Coalition.status == CoalitionStatus.ACTIVE)

        # Order by performance and creation time
        query = query.order_by(desc(Coalition.performance_score), desc(Coalition.created_at))

        result = await session.execute(query)
        coalitions = result.scalars().all()

        query_time = time.time() - start_time
        self._track_query_performance("get_agent_coalitions", query_time)

        return list(coalitions)

    async def get_coalition_members_optimized(
        self,
        session: AsyncSession,
        coalition_id: str,
        include_performance: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get coalition members with role and performance data."""
        start_time = time.time()

        # Optimized query with association table data
        query = (
            select(
                Agent.id,
                Agent.name,
                Agent.status,
                Agent.last_active,
                Agent.inference_count,
                Agent.total_steps,
                agent_coalition_association.c.role,
                agent_coalition_association.c.joined_at,
                agent_coalition_association.c.contribution_score,
                agent_coalition_association.c.trust_score,
            )
            .join(
                agent_coalition_association,
                Agent.id == agent_coalition_association.c.agent_id,
            )
            .where(agent_coalition_association.c.coalition_id == coalition_id)
        )

        if include_performance:
            # Order by contribution and trust scores
            query = query.order_by(
                desc(agent_coalition_association.c.contribution_score),
                desc(agent_coalition_association.c.trust_score),
            )
        else:
            # Simple ordering by join time
            query = query.order_by(agent_coalition_association.c.joined_at)

        result = await session.execute(query)
        members = []

        for row in result:
            members.append(
                {
                    "agent_id": str(row.id),
                    "name": row.name,
                    "status": row.status.value,
                    "last_active": (row.last_active.isoformat() if row.last_active else None),
                    "inference_count": row.inference_count,
                    "total_steps": row.total_steps,
                    "role": row.role.value,
                    "joined_at": row.joined_at.isoformat(),
                    "contribution_score": row.contribution_score,
                    "trust_score": row.trust_score,
                }
            )

        query_time = time.time() - start_time
        self._track_query_performance("get_coalition_members", query_time)

        return members

    async def get_top_performing_agents(
        self,
        session: AsyncSession,
        limit: int = 10,
        metric: str = "inference_count",
    ) -> List[Agent]:
        """Get top performing agents by specified metric."""
        start_time = time.time()

        # Use index on metric fields
        if metric == "inference_count":
            query = (
                select(Agent)
                .where(Agent.status == AgentStatus.ACTIVE)
                .order_by(desc(Agent.inference_count))
            )
        elif metric == "total_steps":
            query = (
                select(Agent)
                .where(Agent.status == AgentStatus.ACTIVE)
                .order_by(desc(Agent.total_steps))
            )
        else:
            # Fallback to creation time
            query = (
                select(Agent)
                .where(Agent.status == AgentStatus.ACTIVE)
                .order_by(desc(Agent.created_at))
            )

        query = query.limit(limit)

        result = await session.execute(query)
        agents = result.scalars().all()

        query_time = time.time() - start_time
        self._track_query_performance("get_top_performing_agents", query_time)

        return list(agents)

    async def get_coalition_performance_stats(
        self, session: AsyncSession, coalition_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive coalition performance statistics."""
        start_time = time.time()

        # Single query to get all stats
        query = (
            select(
                func.count(agent_coalition_association.c.agent_id).label("member_count"),
                func.avg(agent_coalition_association.c.contribution_score).label(
                    "avg_contribution"
                ),
                func.avg(agent_coalition_association.c.trust_score).label("avg_trust"),
                func.max(agent_coalition_association.c.contribution_score).label(
                    "max_contribution"
                ),
                func.min(agent_coalition_association.c.contribution_score).label(
                    "min_contribution"
                ),
                func.sum(Agent.inference_count).label("total_inferences"),
                func.sum(Agent.total_steps).label("total_steps"),
                func.count(Agent.id.distinct())
                .filter(Agent.status == AgentStatus.ACTIVE)
                .label("active_members"),
            )
            .select_from(
                agent_coalition_association.join(
                    Agent, Agent.id == agent_coalition_association.c.agent_id
                )
            )
            .where(agent_coalition_association.c.coalition_id == coalition_id)
        )

        result = await session.execute(query)
        row = result.first()

        if row:
            stats = {
                "member_count": row.member_count or 0,
                "active_members": row.active_members or 0,
                "avg_contribution_score": float(row.avg_contribution or 0),
                "avg_trust_score": float(row.avg_trust or 0),
                "max_contribution_score": float(row.max_contribution or 0),
                "min_contribution_score": float(row.min_contribution or 0),
                "total_inferences": row.total_inferences or 0,
                "total_steps": row.total_steps or 0,
                "activity_ratio": (row.active_members or 0) / max(row.member_count or 1, 1),
            }
        else:
            stats = {
                "member_count": 0,
                "active_members": 0,
                "avg_contribution_score": 0.0,
                "avg_trust_score": 0.0,
                "max_contribution_score": 0.0,
                "min_contribution_score": 0.0,
                "total_inferences": 0,
                "total_steps": 0,
                "activity_ratio": 0.0,
            }

        query_time = time.time() - start_time
        self._track_query_performance("get_coalition_performance_stats", query_time)

        return stats

    async def bulk_update_agent_activity(
        self, session: AsyncSession, agent_updates: List[Dict[str, Any]]
    ) -> int:
        """Bulk update agent activity with optimized query."""
        start_time = time.time()

        if not agent_updates:
            return 0

        # Use bulk update for better performance
        updated_count = 0

        for update_data in agent_updates:
            agent_id = update_data["agent_id"]

            query = (
                update(Agent)
                .where(Agent.id == agent_id)
                .values(
                    last_active=update_data.get("last_active", func.now()),
                    inference_count=Agent.inference_count
                    + update_data.get("inference_increment", 0),
                    total_steps=Agent.total_steps + update_data.get("steps_increment", 0),
                    updated_at=func.now(),
                )
            )

            result = await session.execute(query)
            updated_count += result.rowcount

        await session.commit()

        query_time = time.time() - start_time
        self._track_query_performance("bulk_update_agent_activity", query_time)

        return updated_count

    async def search_agents_optimized(
        self, session: AsyncSession, search_term: str, limit: int = 50
    ) -> List[Agent]:
        """Search agents using full-text search with trigram indexes."""
        start_time = time.time()

        # Use trigram search for better performance
        query = (
            select(Agent)
            .where(
                or_(
                    Agent.name.op("%")(search_term),  # Trigram similarity
                    Agent.template.ilike(f"%{search_term}%"),
                )
            )
            .order_by(
                # Order by similarity score
                func.similarity(Agent.name, search_term).desc(),
                desc(Agent.created_at),
            )
            .limit(limit)
        )

        result = await session.execute(query)
        agents = result.scalars().all()

        query_time = time.time() - start_time
        self._track_query_performance("search_agents", query_time)

        return list(agents)

    async def get_system_performance_metrics(self, session: AsyncSession) -> Dict[str, Any]:
        """Get system-wide performance metrics."""
        start_time = time.time()

        # Single query for multiple metrics
        query = select(
            func.count(Agent.id).label("total_agents"),
            func.count(Agent.id).filter(Agent.status == AgentStatus.ACTIVE).label("active_agents"),
            func.count(Coalition.id).label("total_coalitions"),
            func.count(Coalition.id)
            .filter(Coalition.status == CoalitionStatus.ACTIVE)
            .label("active_coalitions"),
            func.avg(Agent.inference_count).label("avg_inferences_per_agent"),
            func.sum(Agent.inference_count).label("total_inferences"),
            func.sum(Agent.total_steps).label("total_steps"),
            func.avg(Coalition.performance_score).label("avg_coalition_performance"),
            func.count(agent_coalition_association.c.agent_id).label("total_memberships"),
        ).select_from(Agent.outerjoin(Coalition).outerjoin(agent_coalition_association))

        result = await session.execute(query)
        row = result.first()

        metrics = {
            "total_agents": row.total_agents or 0,
            "active_agents": row.active_agents or 0,
            "total_coalitions": row.total_coalitions or 0,
            "active_coalitions": row.active_coalitions or 0,
            "avg_inferences_per_agent": float(row.avg_inferences_per_agent or 0),
            "total_inferences": row.total_inferences or 0,
            "total_steps": row.total_steps or 0,
            "avg_coalition_performance": float(row.avg_coalition_performance or 0),
            "total_memberships": row.total_memberships or 0,
            "agent_utilization": (row.active_agents or 0) / max(row.total_agents or 1, 1),
            "coalition_utilization": (row.active_coalitions or 0)
            / max(row.total_coalitions or 1, 1),
        }

        query_time = time.time() - start_time
        self._track_query_performance("get_system_performance_metrics", query_time)

        return metrics

    def _track_query_performance(self, query_name: str, query_time: float):
        """Track query performance metrics."""
        self.query_stats["total_queries"] += 1

        # Update average query time
        current_avg = self.query_stats["avg_query_time"]
        total_queries = self.query_stats["total_queries"]
        self.query_stats["avg_query_time"] = (
            current_avg * (total_queries - 1) + query_time
        ) / total_queries

        # Track slow queries (>100ms)
        if query_time > 0.1:
            self.query_stats["slow_queries"].append(
                {
                    "query_name": query_name,
                    "query_time": query_time,
                    "timestamp": time.time(),
                }
            )

            # Keep only last 50 slow queries
            if len(self.query_stats["slow_queries"]) > 50:
                self.query_stats["slow_queries"].pop(0)

            logger.warning(f"Slow query detected: {query_name} took {query_time:.3f}s")

    def get_query_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive query performance report."""
        return {
            "total_queries": self.query_stats["total_queries"],
            "avg_query_time": self.query_stats["avg_query_time"],
            "slow_queries_count": len(self.query_stats["slow_queries"]),
            "slow_queries": self.query_stats["slow_queries"][-10:],  # Last 10 slow queries
            "cache_hit_rate": (
                self.query_stats["cache_hits"] / max(self.query_stats["total_queries"], 1)
            )
            * 100,
        }


# Global query optimizer instance
_query_optimizer: Optional[QueryOptimizer] = None


def get_query_optimizer() -> QueryOptimizer:
    """Get global query optimizer instance."""
    global _query_optimizer

    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer()

    return _query_optimizer


# Query result caching decorator
def cache_query(cache_key: str, ttl: int = 300):
    """Decorator to cache query results."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            optimizer = get_query_optimizer()

            # Check cache
            cache_entry = optimizer.query_cache.get(cache_key)
            if cache_entry and (time.time() - cache_entry["timestamp"]) < ttl:
                optimizer.query_stats["cache_hits"] += 1
                return cache_entry["data"]

            # Execute query
            result = await func(*args, **kwargs)

            # Cache result
            optimizer.query_cache[cache_key] = {
                "data": result,
                "timestamp": time.time(),
            }

            optimizer.query_stats["cache_misses"] += 1
            return result

        return wrapper

    return decorator
