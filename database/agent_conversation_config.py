"""
Agent Conversation Database Configuration

This module provides specialized configuration for agent conversation database
operations as specified in Task 39.5. It enhances the base database configuration
with conversation-specific optimizations and monitoring.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import Pool

from database.session import db_state, engine


@dataclass
class AgentConversationDbConfig:
    """Configuration for agent conversation database operations."""

    # Connection pool settings specifically for conversation workloads
    conversation_pool_size: int = int(os.getenv("AGENT_CONV_POOL_SIZE", "10"))
    conversation_max_overflow: int = int(os.getenv("AGENT_CONV_MAX_OVERFLOW", "20"))

    # Query timeout settings
    conversation_query_timeout: int = int(os.getenv("AGENT_CONV_QUERY_TIMEOUT", "30"))
    message_batch_size: int = int(os.getenv("AGENT_CONV_MESSAGE_BATCH_SIZE", "100"))

    # Performance monitoring
    enable_query_logging: bool = os.getenv("AGENT_CONV_QUERY_LOGGING", "false").lower() == "true"
    enable_performance_metrics: bool = (
        os.getenv("AGENT_CONV_PERF_METRICS", "true").lower() == "true"
    )

    # Cache settings
    enable_query_cache: bool = os.getenv("AGENT_CONV_QUERY_CACHE", "true").lower() == "true"
    cache_ttl_seconds: int = int(os.getenv("AGENT_CONV_CACHE_TTL", "300"))  # 5 minutes

    # Conversation limits
    max_messages_per_conversation: int = int(os.getenv("MAX_MESSAGES_PER_CONV", "1000"))
    max_agents_per_conversation: int = int(os.getenv("MAX_AGENTS_PER_CONV", "10"))
    conversation_cleanup_interval_hours: int = int(os.getenv("CONV_CLEANUP_INTERVAL", "24"))


# Global configuration instance
agent_conversation_config = AgentConversationDbConfig()


class AgentConversationDbMonitor:
    """Monitor for agent conversation database operations."""

    def __init__(self):
        self.metrics = {
            "conversations_created": 0,
            "messages_created": 0,
            "query_count": 0,
            "slow_queries": 0,
            "connection_errors": 0,
            "performance_warnings": [],
        }

        # Set up SQLAlchemy event listeners if monitoring is enabled
        if agent_conversation_config.enable_performance_metrics and engine:
            self._setup_event_listeners()

    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners for monitoring."""

        @event.listens_for(Engine, "before_cursor_execute")
        def receive_before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            """Track query start time."""
            context._query_start_time = db_state.last_check_time

        @event.listens_for(Engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Track query completion and performance."""
            total_time = db_state.last_check_time - context._query_start_time
            self.metrics["query_count"] += 1

            # Log slow queries (>1 second)
            if total_time > 1.0:
                self.metrics["slow_queries"] += 1
                if agent_conversation_config.enable_query_logging:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Slow agent conversation query ({total_time:.2f}s): {statement[:200]}..."
                    )

        @event.listens_for(Pool, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """Track successful connections."""
            db_state.record_success()

        @event.listens_for(Pool, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Monitor connection pool usage."""
            if hasattr(engine, "pool"):
                pool = engine.pool
                checked_out = pool.checkedout()
                pool_size = pool.size()

                # Warn if pool utilization is high
                if checked_out / max(pool_size, 1) > 0.8:
                    self.metrics["performance_warnings"].append(
                        {
                            "type": "high_pool_utilization",
                            "checked_out": checked_out,
                            "pool_size": pool_size,
                            "utilization": checked_out / pool_size,
                            "timestamp": db_state.last_check_time,
                        }
                    )

    def record_conversation_created(self):
        """Record that a conversation was created."""
        self.metrics["conversations_created"] += 1

    def record_message_created(self):
        """Record that a message was created."""
        self.metrics["messages_created"] += 1

    def record_connection_error(self):
        """Record a connection error."""
        self.metrics["connection_errors"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            **self.metrics,
            "config": {
                "pool_size": agent_conversation_config.conversation_pool_size,
                "max_overflow": agent_conversation_config.conversation_max_overflow,
                "query_timeout": agent_conversation_config.conversation_query_timeout,
                "query_logging": agent_conversation_config.enable_query_logging,
                "performance_metrics": agent_conversation_config.enable_performance_metrics,
            },
            "database_state": {
                "is_available": db_state.is_available,
                "error_count": db_state.error_count,
                "consecutive_failures": db_state.consecutive_failures,
                "last_error": db_state.last_error,
            },
        }

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            "conversations_created": 0,
            "messages_created": 0,
            "query_count": 0,
            "slow_queries": 0,
            "connection_errors": 0,
            "performance_warnings": [],
        }


# Global monitor instance
agent_conversation_monitor = AgentConversationDbMonitor()


def get_agent_conversation_db_health() -> Dict[str, Any]:
    """Get health status specific to agent conversation database operations."""

    from database.session import check_database_health

    base_health = check_database_health()
    conversation_metrics = agent_conversation_monitor.get_metrics()

    # Calculate health score based on metrics
    health_score = 100.0

    if conversation_metrics["connection_errors"] > 0:
        health_score -= min(conversation_metrics["connection_errors"] * 10, 50)

    if conversation_metrics["slow_queries"] > 0:
        slow_query_ratio = conversation_metrics["slow_queries"] / max(
            conversation_metrics["query_count"], 1
        )
        health_score -= slow_query_ratio * 30

    if len(conversation_metrics["performance_warnings"]) > 0:
        health_score -= len(conversation_metrics["performance_warnings"]) * 5

    health_score = max(0, health_score)

    return {
        **base_health,
        "conversation_specific": {
            "health_score": health_score,
            "status": "healthy"
            if health_score > 80
            else "degraded"
            if health_score > 50
            else "unhealthy",
            "metrics": conversation_metrics,
            "recommendations": _get_performance_recommendations(conversation_metrics),
        },
    }


def _get_performance_recommendations(metrics: Dict[str, Any]) -> List[str]:
    """Get performance recommendations based on current metrics."""

    recommendations = []

    if metrics["slow_queries"] > metrics["query_count"] * 0.1:
        recommendations.append(
            "Consider adding database indexes for frequently queried conversation data"
        )

    if metrics["connection_errors"] > 0:
        recommendations.append("Check database connection stability and network connectivity")

    if len(metrics["performance_warnings"]) > 0:
        for warning in metrics["performance_warnings"][-3:]:  # Show last 3 warnings
            if warning["type"] == "high_pool_utilization":
                recommendations.append(
                    f"Consider increasing connection pool size (current: {warning['pool_size']}, "
                    f"utilization: {warning['utilization']:.1%})"
                )

    if metrics["conversations_created"] > 1000 and not agent_conversation_config.enable_query_cache:
        recommendations.append("Consider enabling query caching for better performance")

    return recommendations


def optimize_agent_conversation_queries():
    """Apply query optimizations for agent conversation operations."""

    if not engine:
        return

    try:
        with engine.connect() as conn:
            # Set statement timeout for conversation queries
            if agent_conversation_config.conversation_query_timeout > 0:
                conn.execute(
                    f"SET statement_timeout = {agent_conversation_config.conversation_query_timeout * 1000}"
                )

            # Enable query plan caching if supported
            if "postgresql" in str(engine.url):
                conn.execute("SET shared_preload_libraries = 'pg_stat_statements'")
                conn.execute("SET track_activity_query_size = 2048")

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Could not apply conversation query optimizations: {e}")


def validate_agent_conversation_config() -> Dict[str, Any]:
    """Validate the agent conversation database configuration."""

    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "config": agent_conversation_config.__dict__,
    }

    # Validate pool sizes
    if agent_conversation_config.conversation_pool_size < 5:
        validation_results["warnings"].append(
            "Pool size is very low, may cause connection bottlenecks"
        )

    if agent_conversation_config.conversation_pool_size > 50:
        validation_results["warnings"].append(
            "Pool size is very high, may consume excessive resources"
        )

    # Validate timeouts
    if agent_conversation_config.conversation_query_timeout < 10:
        validation_results["warnings"].append(
            "Query timeout is very low, may cause premature query cancellation"
        )

    # Validate message limits
    if agent_conversation_config.max_messages_per_conversation > 10000:
        validation_results["warnings"].append(
            "Maximum messages per conversation is very high, may impact performance"
        )

    if agent_conversation_config.max_agents_per_conversation > 20:
        validation_results["warnings"].append(
            "Maximum agents per conversation is very high, may impact performance"
        )

    # Check if cache TTL is reasonable
    if (
        agent_conversation_config.enable_query_cache
        and agent_conversation_config.cache_ttl_seconds < 60
    ):
        validation_results["warnings"].append("Cache TTL is very low, caching may not be effective")

    # Validate environment-specific settings
    if os.getenv("PRODUCTION", "false").lower() == "true":
        if not agent_conversation_config.enable_performance_metrics:
            validation_results["warnings"].append("Performance metrics disabled in production")

        if agent_conversation_config.enable_query_logging:
            validation_results["warnings"].append(
                "Query logging enabled in production, may impact performance"
            )

    return validation_results


# Initialize optimizations on module import
if engine:
    try:
        optimize_agent_conversation_queries()
    except Exception:
        pass  # Ignore errors during initialization
