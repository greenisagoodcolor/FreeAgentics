"""
FreeAgentics Test Reporting System

This module provides comprehensive test reporting, metrics collection, and coverage analysis
for the FreeAgentics project. It includes:

- Coverage analysis and reporting
- Test metrics collection and flaky test detection
- Comprehensive dashboard generation
- Report archival and cleanup system
- Integration with pytest for automatic metrics collection
"""

from .coverage_analyzer import CoverageAnalyzer, CoverageReport, CoverageStats
from .dashboard_generator import DashboardGenerator
from .report_archival_system import ArchivalConfig, ReportArchivalSystem, RetentionPolicy
from .test_metrics_collector import TestMetric, TestMetricsCollector, TestStatus, TestSuiteMetrics
from .test_reporting_integration import TestReportingIntegration

__version__ = "1.0.0"
__author__ = "FreeAgentics Team"

__all__ = [
    # Core classes
    "CoverageAnalyzer",
    "TestMetricsCollector",
    "DashboardGenerator",
    "ReportArchivalSystem",
    "TestReportingIntegration",
    # Data classes
    "CoverageStats",
    "CoverageReport",
    "TestMetric",
    "TestSuiteMetrics",
    "ArchivalConfig",
    # Enums
    "TestStatus",
    "RetentionPolicy",
]


# Convenience functions
def generate_all_reports(test_run_id: str = None, output_dir: str = "tests/reporting"):
    """Generate all reports using the integration system."""
    integration = TestReportingIntegration(output_dir)
    return integration.run_comprehensive_reporting(test_run_id)


def run_health_check(output_dir: str = "tests/reporting"):
    """Run health check on the reporting system."""
    integration = TestReportingIntegration(output_dir)
    return integration.run_health_check()


def cleanup_old_reports(days: int = 30):
    """Clean up old reports and data."""
    archival_system = ReportArchivalSystem()
    return archival_system.run_archival_process()


# Module-level configuration
DEFAULT_CONFIG = {
    "coverage_threshold": 80.0,
    "flaky_test_threshold": 10.0,
    "slow_test_threshold": 1.0,
    "retention_days": 30,
    "compression_days": 7,
    "generate_html": True,
    "generate_json": True,
    "auto_archive": True,
}
