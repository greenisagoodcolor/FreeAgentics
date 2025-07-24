"""PostgreSQL test infrastructure for FreeAgentics load testing."""

from .data_generator import TestDataGenerator
from .db_reset import DatabaseReset
from .load_test import DatabaseLoadTest
from .performance_monitor import LoadTestRunner, PerformanceMonitor
from .pool_config import DatabasePool, PerformancePool, close_all_pools, get_pool

__all__ = [
    "DatabasePool",
    "PerformancePool",
    "get_pool",
    "close_all_pools",
    "TestDataGenerator",
    "DatabaseReset",
    "PerformanceMonitor",
    "LoadTestRunner",
    "DatabaseLoadTest",
]

__version__ = "1.0.0"
