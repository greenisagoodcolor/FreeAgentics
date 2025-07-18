"""
Extended health check endpoints with detailed system diagnostics.

Provides comprehensive health information for monitoring systems.
"""

import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from agents.agent_manager import AgentManager
from database.session import get_db
from observability.performance_metrics import performance_tracker

router = APIRouter()


class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
        self.check_history = {}
        self.startup_time = time.time()

    async def check_database(self, db: Session) -> Dict[str, Any]:
        """Check database health and performance."""
        start_time = time.time()

        try:
            # Basic connectivity check
            result = db.execute(text("SELECT 1"))
            result.fetchone()

            # Connection pool status
            pool_status = db.execute(
                text(
                    """
                    SELECT count(*) as total_connections,
                           count(*) FILTER (WHERE state = 'active') as active,
                           count(*) FILTER (WHERE state = 'idle') as idle,
                           count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                """
                )
            ).fetchone()

            # Database size
            db_size = db.execute(
                text("SELECT pg_database_size(current_database()) as size")
            ).fetchone()

            # Slow query check
            slow_queries = (
                db.execute(
                    text(
                        """
                    SELECT count(*) as count
                    FROM pg_stat_statements
                    WHERE mean_exec_time > 100
                    AND query NOT LIKE '%pg_stat%'
                    LIMIT 1
                """
                    )
                ).fetchone()
                if self._has_pg_stat_statements(db)
                else None
            )

            latency = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "connections": {
                    "total": pool_status.total_connections
                    if pool_status
                    else 0,
                    "active": pool_status.active if pool_status else 0,
                    "idle": pool_status.idle if pool_status else 0,
                    "idle_in_transaction": pool_status.idle_in_transaction
                    if pool_status
                    else 0,
                },
                "database_size_mb": round(db_size.size / 1024 / 1024, 2)
                if db_size
                else 0,
                "slow_queries": slow_queries.count if slow_queries else 0,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": round((time.time() - start_time) * 1000, 2),
            }

    def _has_pg_stat_statements(self, db: Session) -> bool:
        """Check if pg_stat_statements extension is available."""
        try:
            result = db.execute(
                text(
                    "SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'"
                )
            ).fetchone()
            return result is not None
        except:
            return False

    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis health and performance."""
        try:
            import redis

            start_time = time.time()
            r = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"), port=6379
            )

            # Ping check
            r.ping()

            # Get Redis info
            info = r.info()
            memory_info = r.info("memory")

            latency = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": round(
                    memory_info.get("used_memory", 0) / 1024 / 1024, 2
                ),
                "memory_usage_ratio": round(
                    memory_info.get("used_memory", 0)
                    / memory_info.get("maxmemory", 1),
                    3,
                )
                if memory_info.get("maxmemory", 0) > 0
                else 0,
                "hit_rate": self._calculate_hit_rate(info),
                "evicted_keys": info.get("evicted_keys", 0),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate Redis cache hit rate."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return round(hits / total, 3) if total > 0 else 0

    async def check_agents(self) -> Dict[str, Any]:
        """Check agent system health."""
        try:
            from agents.agent_manager import AgentManager

            # Get agent manager instance
            manager = AgentManager()

            # Count active agents
            active_agents = len(manager.agents)

            # Get performance metrics
            agent_metrics = []
            for agent_id, agent in manager.agents.items():
                metrics = {
                    "agent_id": agent_id,
                    "status": "active",
                    "memory_mb": 0,  # Would need actual memory tracking
                    "inference_count": 0,  # Would need actual counting
                    "error_count": 0,  # Would need actual error tracking
                }
                agent_metrics.append(metrics)

            return {
                "status": "healthy" if active_agents > 0 else "idle",
                "active_agents": active_agents,
                "total_capacity": 50,  # Configurable max agents
                "utilization": round(active_agents / 50, 3),
                "agents": agent_metrics[:5],  # Top 5 agents
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "active_agents": 0}

    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()

            # Memory usage
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage('/')

            # Network I/O
            net_io = psutil.net_io_counters()

            # Process info
            process = psutil.Process()
            process_info = {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections()),
            }

            return {
                "status": "healthy",
                "cpu": {
                    "usage_percent": cpu_percent,
                    "cores": cpu_count,
                    "load_average": os.getloadavg(),
                },
                "memory": {
                    "total_mb": round(memory.total / 1024 / 1024, 2),
                    "used_mb": round(memory.used / 1024 / 1024, 2),
                    "available_mb": round(memory.available / 1024 / 1024, 2),
                    "usage_percent": memory.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
                    "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                    "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
                    "usage_percent": disk.percent,
                },
                "network": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "err_in": net_io.errin,
                    "err_out": net_io.errout,
                },
                "process": process_info,
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def calculate_overall_health(self, checks: Dict[str, Dict]) -> str:
        """Calculate overall system health status."""
        statuses = [
            check.get("status", "unknown") for check in checks.values()
        ]

        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"


# Global health checker instance
health_checker = HealthChecker()


@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """
    Comprehensive health check with detailed diagnostics.

    Returns detailed information about all system components.
    """
    start_time = time.time()

    # Run all health checks
    checks = {
        "database": await health_checker.check_database(db),
        "redis": await health_checker.check_redis(),
        "agents": await health_checker.check_agents(),
        "system": await health_checker.check_system_resources(),
    }

    # Calculate overall health
    overall_status = health_checker.calculate_overall_health(checks)

    # Get uptime
    uptime_seconds = time.time() - health_checker.startup_time

    response = {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": round(uptime_seconds, 2),
        "uptime_human": str(timedelta(seconds=int(uptime_seconds))),
        "version": os.getenv("FREEAGENTICS_VERSION", "0.0.1-dev"),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "checks": checks,
        "total_check_time_ms": round((time.time() - start_time) * 1000, 2),
    }

    # Return appropriate status code
    status_code = 200 if overall_status == "healthy" else 503

    return JSONResponse(content=response, status_code=status_code)


@router.get("/health/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """
    Kubernetes readiness probe endpoint.

    Checks if the service is ready to accept traffic.
    """
    try:
        # Check database
        db.execute(text("SELECT 1"))

        # Check if we have minimum agents available
        from agents.agent_manager import AgentManager

        manager = AgentManager()

        if (
            len(manager.agents) == 0
            and time.time() - health_checker.startup_time > 60
        ):
            # No agents after 1 minute of startup
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "reason": "no_agents_available",
                },
                status_code=503,
            )

        return {"status": "ready"}

    except Exception as e:
        return JSONResponse(
            content={"status": "not_ready", "error": str(e)}, status_code=503
        )


@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.

    Simple check to verify the service is alive.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/startup")
async def startup_check(db: Session = Depends(get_db)):
    """
    Kubernetes startup probe endpoint.

    Checks if the application has completed startup.
    """
    try:
        # Check database migrations
        result = db.execute(
            text("SELECT 1 FROM alembic_version LIMIT 1")
        ).fetchone()

        if not result:
            return JSONResponse(
                content={
                    "status": "starting",
                    "reason": "database_migrations_pending",
                },
                status_code=503,
            )

        # Check if core services are initialized
        startup_duration = time.time() - health_checker.startup_time

        if startup_duration < 10:  # Give 10 seconds for startup
            return JSONResponse(
                content={
                    "status": "starting",
                    "reason": "initializing_services",
                    "startup_duration": round(startup_duration, 2),
                },
                status_code=503,
            )

        return {
            "status": "started",
            "startup_duration": round(startup_duration, 2),
        }

    except Exception as e:
        return JSONResponse(
            content={"status": "starting", "error": str(e)}, status_code=503
        )


@router.get("/health/dependencies")
async def dependency_health_check():
    """
    Check health of all external dependencies.

    Useful for debugging integration issues.
    """
    dependencies = {}

    # Check external services
    external_checks = [
        ("postgresql", check_postgresql_external),
        ("redis", check_redis_external),
        ("elasticsearch", check_elasticsearch_external),
        ("prometheus", check_prometheus_external),
    ]

    for name, check_func in external_checks:
        dependencies[name] = await check_func()

    # Calculate overall dependency health
    all_healthy = all(
        dep.get("status") == "healthy" for dep in dependencies.values()
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "dependencies": dependencies,
    }


async def check_postgresql_external() -> Dict[str, Any]:
    """Check external PostgreSQL health."""
    try:
        import asyncpg

        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            user=os.getenv("POSTGRES_USER", "freeagentics"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB", "freeagentics"),
        )

        version = await conn.fetchval("SELECT version()")
        await conn.close()

        return {
            "status": "healthy",
            "version": version.split()[1] if version else "unknown",
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_redis_external() -> Dict[str, Any]:
    """Check external Redis health."""
    try:
        import aioredis

        redis = await aioredis.create_redis_pool(
            f"redis://{os.getenv('REDIS_HOST', 'localhost')}:6379"
        )

        await redis.ping()
        info = await redis.info()
        redis.close()
        await redis.wait_closed()

        return {
            "status": "healthy",
            "version": info.get("redis_version", "unknown"),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_elasticsearch_external() -> Dict[str, Any]:
    """Check external Elasticsearch health."""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = f"http://{os.getenv('ELASTICSEARCH_HOST', 'localhost')}:9200/_cluster/health"
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "status": "healthy"
                        if data.get("status") != "red"
                        else "unhealthy",
                        "cluster_status": data.get("status", "unknown"),
                    }
        return {"status": "unhealthy", "error": "Unable to connect"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_prometheus_external() -> Dict[str, Any]:
    """Check external Prometheus health."""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = f"http://{os.getenv('PROMETHEUS_HOST', 'localhost')}:9090/-/healthy"
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    return {"status": "healthy"}
        return {"status": "unhealthy", "error": "Unable to connect"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
