"""
Performance Integration Module for FreeAgentics.

This module provides a unified interface to enable all performance optimizations
in the FreeAgentics system. It integrates:

1. Performance monitoring and alerting
2. Threading optimizations with adaptive pools
3. Database connection pooling and query optimization
4. API response caching and compression
5. Memory management and garbage collection tuning
6. Comprehensive benchmarking and validation

Usage:
    from performance_integration import PerformanceManager

    manager = PerformanceManager()
    await manager.initialize_all_optimizations()

    # System is now fully optimized
    app = manager.get_optimized_app()
"""

import asyncio
import logging
import os
from typing import Any, Dict

from fastapi import FastAPI

# Import all optimization modules
from agents.optimized_agent_manager import (
    OptimizationConfig,
    OptimizedAgentManager,
)
from api.performance_middleware import (
    PerformanceConfig,
    setup_performance_middleware,
)
from benchmarks.performance_benchmark_suite import PerformanceBenchmarkRunner
from database.optimized_db import DatabaseConfig, initialize_optimized_db
from observability.memory_optimizer import (
    get_memory_optimizer,
    start_memory_optimization,
)
from observability.performance_monitor import (
    get_performance_monitor,
    start_performance_monitoring,
)

logger = logging.getLogger(__name__)


class PerformanceManager:
    """Unified performance optimization manager for FreeAgentics."""

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.optimizations_enabled = {}
        self.performance_monitor = None
        self.memory_optimizer = None
        self.agent_manager = None
        self.benchmark_runner = None
        self.app = None

        # Performance configurations
        self.configs = self._create_default_configs()

        logger.info(f"Performance manager initialized for {environment} environment")

    def _create_default_configs(self) -> Dict[str, Any]:
        """Create default configurations for different environments."""

        base_configs = {
            "threading": OptimizationConfig(
                cpu_aware_sizing=True,
                work_stealing_enabled=True,
                batch_size=20,
                batch_timeout_ms=50,
                memory_pooling_enabled=True,
                gil_aware_scheduling=True,
                async_io_enabled=True,
            ),
            "database": DatabaseConfig(
                min_connections=5,
                max_connections=50,
                query_cache_size=1000,
                auto_scaling_enabled=True,
                slow_query_threshold=0.1,
                health_check_interval=30.0,
            ),
            "api": PerformanceConfig(
                caching_enabled=True,
                compression_enabled=True,
                deduplication_enabled=True,
                monitoring_enabled=True,
                slow_request_threshold=1.0,
            ),
        }

        # Environment-specific adjustments
        if self.environment == "development":
            # More aggressive monitoring for development
            base_configs["database"].slow_query_threshold = 0.05
            base_configs["api"].slow_request_threshold = 0.5

        elif self.environment == "production":
            # Production-optimized settings
            base_configs["threading"].batch_size = 50
            base_configs["database"].max_connections = 100
            base_configs["api"].cache_config.max_size = 2000

        elif self.environment == "testing":
            # Testing-friendly settings
            base_configs["threading"].batch_size = 10
            base_configs["database"].min_connections = 2
            base_configs["database"].max_connections = 10

        return base_configs

    async def initialize_all_optimizations(self):
        """Initialize all performance optimizations."""
        logger.info("Starting comprehensive performance optimization initialization")

        try:
            # 1. Start performance monitoring
            await self._initialize_monitoring()

            # 2. Initialize memory optimization
            await self._initialize_memory_optimization()

            # 3. Initialize database optimizations
            await self._initialize_database_optimization()

            # 4. Initialize threading optimizations
            await self._initialize_threading_optimization()

            # 5. Initialize API optimizations
            await self._initialize_api_optimization()

            # 6. Initialize benchmarking
            await self._initialize_benchmarking()

            # 7. Validate all optimizations
            await self._validate_optimizations()

            logger.info("All performance optimizations initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize optimizations: {e}")
            raise

    async def _initialize_monitoring(self):
        """Initialize performance monitoring."""
        logger.info("Initializing performance monitoring")

        start_performance_monitoring()
        self.performance_monitor = get_performance_monitor()

        self.optimizations_enabled["monitoring"] = True
        logger.info("Performance monitoring initialized")

    async def _initialize_memory_optimization(self):
        """Initialize memory optimization."""
        logger.info("Initializing memory optimization")

        # Determine workload type based on environment
        workload_type = {
            "development": "mixed",
            "production": "long_running",
            "testing": "high_allocation",
        }.get(self.environment, "mixed")

        start_memory_optimization(workload_type)
        self.memory_optimizer = get_memory_optimizer()

        self.optimizations_enabled["memory"] = True
        logger.info(f"Memory optimization initialized for {workload_type} workload")

    async def _initialize_database_optimization(self):
        """Initialize database optimization."""
        logger.info("Initializing database optimization")

        # Configure database based on environment
        config = self.configs["database"]

        # Set database URL (in production, get from environment)
        database_url = os.getenv(
            "DATABASE_URL", "postgresql://user:pass@localhost/freeagentics"
        )
        config.write_primary_url = database_url

        # Add read replicas if available
        read_replica_urls = os.getenv("READ_REPLICA_URLS", "").split(",")
        if read_replica_urls and read_replica_urls[0]:
            config.read_replica_urls = read_replica_urls

        try:
            await initialize_optimized_db(config)
            self.optimizations_enabled["database"] = True
            logger.info("Database optimization initialized")
        except Exception as e:
            logger.warning(f"Database optimization failed: {e}")
            # Continue without database optimization
            self.optimizations_enabled["database"] = False

    async def _initialize_threading_optimization(self):
        """Initialize threading optimization."""
        logger.info("Initializing threading optimization")

        config = self.configs["threading"]
        self.agent_manager = OptimizedAgentManager(config)

        self.optimizations_enabled["threading"] = True
        logger.info("Threading optimization initialized")

    async def _initialize_api_optimization(self):
        """Initialize API optimization."""
        logger.info("Initializing API optimization")

        # Create FastAPI app
        self.app = FastAPI(
            title="FreeAgentics Optimized API",
            description="High-performance multi-agent system with comprehensive optimizations",
            version="1.0.0",
        )

        # Add performance middleware
        config = self.configs["api"]
        setup_performance_middleware(self.app, config)

        # Add performance monitoring endpoints
        self._add_performance_endpoints()

        self.optimizations_enabled["api"] = True
        logger.info("API optimization initialized")

    async def _initialize_benchmarking(self):
        """Initialize benchmarking suite."""
        logger.info("Initializing benchmarking suite")

        self.benchmark_runner = PerformanceBenchmarkRunner()

        self.optimizations_enabled["benchmarking"] = True
        logger.info("Benchmarking suite initialized")

    def _add_performance_endpoints(self):
        """Add performance monitoring endpoints to FastAPI app."""

        @self.app.get("/performance/status")
        async def get_performance_status():
            """Get current performance status."""
            return {
                "optimizations_enabled": self.optimizations_enabled,
                "environment": self.environment,
                "timestamp": self.performance_monitor.get_current_metrics().timestamp.isoformat(),
            }

        @self.app.get("/performance/metrics")
        async def get_performance_metrics():
            """Get detailed performance metrics."""
            return self.performance_monitor.get_performance_report()

        @self.app.get("/performance/memory")
        async def get_memory_stats():
            """Get memory optimization statistics."""
            return self.memory_optimizer.get_comprehensive_report()

        @self.app.get("/performance/database")
        async def get_database_stats():
            """Get database performance statistics."""
            if self.optimizations_enabled.get("database"):
                from database.optimized_db import get_db_statistics

                return await get_db_statistics()
            return {"error": "Database optimization not enabled"}

        @self.app.get("/performance/threading")
        async def get_threading_stats():
            """Get threading optimization statistics."""
            if self.agent_manager:
                return self.agent_manager.get_statistics()
            return {"error": "Threading optimization not enabled"}

        @self.app.post("/performance/gc")
        async def force_garbage_collection():
            """Force garbage collection."""
            return self.performance_monitor.force_garbage_collection()

        @self.app.post("/performance/optimize")
        async def force_memory_optimization():
            """Force memory optimization."""
            self.memory_optimizer.optimize_memory()
            return {"message": "Memory optimization triggered"}

        @self.app.post("/performance/benchmark/{suite_name}")
        async def run_benchmark_suite(suite_name: str):
            """Run a specific benchmark suite."""
            try:
                results = await self.benchmark_runner.run_benchmark_suite(suite_name)
                return {
                    "suite": suite_name,
                    "results": [
                        {
                            "name": r.name,
                            "throughput": r.throughput_ops_per_second,
                            "duration": r.duration_seconds,
                            "success": r.success,
                        }
                        for r in results
                    ],
                }
            except Exception as e:
                return {"error": str(e)}

    async def _validate_optimizations(self):
        """Validate that all optimizations are working correctly."""
        logger.info("Validating performance optimizations")

        validation_results = {}

        # Check monitoring
        if self.optimizations_enabled["monitoring"]:
            current_metrics = self.performance_monitor.get_current_metrics()
            validation_results["monitoring"] = current_metrics is not None

        # Check memory optimization
        if self.optimizations_enabled["memory"]:
            memory_report = self.memory_optimizer.get_comprehensive_report()
            validation_results["memory"] = memory_report is not None

        # Check threading optimization
        if self.optimizations_enabled["threading"]:
            threading_stats = self.agent_manager.get_statistics()
            validation_results["threading"] = (
                threading_stats["thread_pool"]["workers"] > 0
            )

        # Check database optimization
        if self.optimizations_enabled["database"]:
            try:
                from database.optimized_db import get_db_statistics

                db_stats = await get_db_statistics()
                validation_results["database"] = db_stats is not None
            except Exception as e:
                validation_results["database"] = False
                logger.warning(f"Database validation failed: {e}")

        # Log validation results
        for optimization, is_valid in validation_results.items():
            status = "✓" if is_valid else "✗"
            logger.info(f"{status} {optimization} optimization validation")

        # Check if all enabled optimizations are valid
        failed_optimizations = [
            opt
            for opt, valid in validation_results.items()
            if not valid and self.optimizations_enabled.get(opt, False)
        ]

        if failed_optimizations:
            logger.warning(f"Failed optimizations: {failed_optimizations}")
        else:
            logger.info("All optimizations validated successfully")

        return validation_results

    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        logger.info("Starting comprehensive performance benchmark")

        if not self.benchmark_runner:
            raise RuntimeError("Benchmarking not initialized")

        # Run all benchmark suites
        suite_names = [
            "threading",
            "database",
            "api",
            "memory",
            "agent_coordination",
        ]
        all_results = {}

        for suite_name in suite_names:
            try:
                logger.info(f"Running {suite_name} benchmark suite")
                results = await self.benchmark_runner.run_benchmark_suite(suite_name)
                all_results[suite_name] = results

                # Log summary
                successful = [r for r in results if r.success]
                avg_throughput = (
                    sum(r.throughput_ops_per_second for r in successful)
                    / len(successful)
                    if successful
                    else 0
                )
                logger.info(
                    f"{suite_name} benchmark completed: {len(successful)}/{len(results)} successful, {avg_throughput:.1f} avg ops/sec"
                )

            except Exception as e:
                logger.error(f"Benchmark suite {suite_name} failed: {e}")
                all_results[suite_name] = {"error": str(e)}

        # Generate comprehensive report
        report = self.benchmark_runner.generate_performance_report()

        logger.info("Comprehensive benchmark completed")
        return {
            "suite_results": all_results,
            "comprehensive_report": report,
            "optimizations_status": self.optimizations_enabled,
        }

    def get_optimized_app(self) -> FastAPI:
        """Get the optimized FastAPI application."""
        if not self.app:
            raise RuntimeError("API optimization not initialized")
        return self.app

    def get_optimized_agent_manager(self) -> OptimizedAgentManager:
        """Get the optimized agent manager."""
        if not self.agent_manager:
            raise RuntimeError("Threading optimization not initialized")
        return self.agent_manager

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get the status of all optimizations."""
        return {
            "environment": self.environment,
            "optimizations_enabled": self.optimizations_enabled,
            "configs": {
                "threading": self.configs["threading"].__dict__,
                "database": self.configs["database"].__dict__,
                "api": self.configs["api"].__dict__,
            },
        }

    async def shutdown(self):
        """Shutdown all optimizations gracefully."""
        logger.info("Shutting down performance optimizations")

        # Shutdown agent manager
        if self.agent_manager:
            self.agent_manager.shutdown()

        # Shutdown database connections
        if self.optimizations_enabled.get("database"):
            from database.optimized_db import close_optimized_db

            await close_optimized_db()

        # Stop monitoring
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()

        # Stop memory optimization
        if self.memory_optimizer:
            self.memory_optimizer.stop_optimization()

        logger.info("Performance optimizations shutdown complete")


# Convenience functions for easy integration
async def create_optimized_system(
    environment: str = "production",
) -> PerformanceManager:
    """Create a fully optimized FreeAgentics system."""
    manager = PerformanceManager(environment)
    await manager.initialize_all_optimizations()
    return manager


async def quick_benchmark(environment: str = "production") -> Dict[str, Any]:
    """Run a quick performance benchmark."""
    manager = await create_optimized_system(environment)
    try:
        return await manager.run_comprehensive_benchmark()
    finally:
        await manager.shutdown()


# Example usage
async def main():
    """Example usage of the performance integration."""
    print("=" * 80)
    print("FREEAGENTICS PERFORMANCE INTEGRATION DEMO")
    print("=" * 80)

    # Create optimized system
    manager = await create_optimized_system("development")

    # Get optimization status
    status = manager.get_optimization_status()
    print("\nOptimization Status:")
    for opt, enabled in status["optimizations_enabled"].items():
        print(f"  {opt}: {'✓' if enabled else '✗'}")

    # Run quick benchmark
    print("\nRunning quick benchmark...")
    try:
        benchmark_results = await manager.run_comprehensive_benchmark()

        print("Benchmark Results:")
        for suite, results in benchmark_results["suite_results"].items():
            if isinstance(results, list):
                successful = [r for r in results if r.success]
                print(f"  {suite}: {len(successful)}/{len(results)} successful")

        report = benchmark_results["comprehensive_report"]
        print("\nOverall Performance:")
        print(f"  Success rate: {report['overall_stats']['success_rate']:.1f}%")
        print(
            f"  Average throughput: {report['overall_stats']['avg_throughput']:.1f} ops/sec"
        )

    except Exception as e:
        print(f"Benchmark failed: {e}")

    # Shutdown
    await manager.shutdown()

    print("\n" + "=" * 80)
    print("DEMO COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
