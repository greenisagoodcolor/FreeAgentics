#!/usr/bin/env python3
"""
Production Readiness Validation for FreeAgentics v1.0.0-alpha+
PIPELINE-ARCHITECT: Final validation for Nemesis mandate
"""

import sys
import json
from datetime import datetime
from pathlib import Path

class ProductionReadinessValidator:
    """Comprehensive production readiness validation."""

    def __init__(self):
        self.results = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "pipeline_architect": "NEMESIS_COMPLETION_VALIDATION",
            "version": "1.0.0-alpha+",
            "checks": {},
            "overall_status": "PENDING",
            "blockers": [],
            "warnings": [],
            "recommendations": []
        }

    def check_infrastructure_files(self):
        """Check all critical infrastructure files are present."""
        required_files = [
            "Dockerfile.production",
            "docker-compose.production.yml",
            "requirements-production.txt",
            "pyproject.toml",
            "alembic.ini",
            ".github/workflows/unified-pipeline.yml",
            "README.md"
        ]

        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)

        if missing_files:
            self.results["blockers"].extend([f"Missing critical file: {f}" for f in missing_files])
            status = "FAILED"
        else:
            status = "PASSED"

        self.results["checks"]["infrastructure_files"] = {
            "status": status,
            "required_files": len(required_files),
            "present_files": len(required_files) - len(missing_files),
            "missing_files": missing_files
        }

    def check_security_configuration(self):
        """Validate security configuration."""
        security_checks = []

        # Check for security headers middleware
        if Path("api/middleware/security_headers.py").exists():
            security_checks.append("Security headers middleware present")
        else:
            self.results["warnings"].append("Security headers middleware not found")

        # Check for rate limiting
        if Path("api/middleware/rate_limiter.py").exists():
            security_checks.append("Rate limiting middleware present")
        else:
            self.results["warnings"].append("Rate limiting middleware not found")

        # Check for authentication
        if Path("auth/").exists():
            security_checks.append("Authentication module present")
        else:
            self.results["blockers"].append("Authentication module not found")

        self.results["checks"]["security_configuration"] = {
            "status": "PASSED" if len(security_checks) >= 2 else "WARNING",
            "active_security_features": security_checks,
            "security_score": len(security_checks) * 25  # Out of 100
        }

    def check_observability_setup(self):
        """Check monitoring and observability configuration."""
        observability_checks = []

        # Check for observability module
        if Path("observability/").exists():
            observability_checks.append("Observability module present")

        # Check for monitoring configs
        monitoring_configs = [
            "monitoring/prometheus.yml",
            "monitoring/alertmanager.yml",
            "monitoring/grafana/"
        ]

        for config in monitoring_configs:
            if Path(config).exists():
                observability_checks.append(f"Monitoring config: {config}")

        self.results["checks"]["observability_setup"] = {
            "status": "PASSED" if observability_checks else "WARNING",
            "observability_features": observability_checks
        }

    def check_api_endpoints(self):
        """Validate API endpoints structure."""
        api_modules = [
            "api/v1/health.py",
            "api/v1/agents.py",
            "api/v1/auth.py",
            "api/main.py"
        ]

        present_modules = []
        missing_modules = []

        for module in api_modules:
            if Path(module).exists():
                present_modules.append(module)
            else:
                missing_modules.append(module)

        if "api/main.py" not in present_modules:
            self.results["blockers"].append("Critical API main module missing")

        self.results["checks"]["api_endpoints"] = {
            "status": "PASSED" if len(present_modules) >= len(api_modules) * 0.8 else "FAILED",
            "present_modules": present_modules,
            "missing_modules": missing_modules,
            "api_coverage": f"{len(present_modules)}/{len(api_modules)}"
        }

    def check_database_migrations(self):
        """Check database migration setup."""
        migration_status = "PASSED"
        migration_info = {}

        if not Path("alembic/").exists():
            self.results["blockers"].append("Alembic migration directory missing")
            migration_status = "FAILED"
        else:
            # Count migration files
            migration_files = list(Path("alembic/versions/").glob("*.py"))
            migration_info["migration_files_count"] = len(migration_files)

            if len(migration_files) == 0:
                self.results["warnings"].append("No database migrations found")
                migration_status = "WARNING"

        self.results["checks"]["database_migrations"] = {
            "status": migration_status,
            **migration_info
        }

    def check_test_coverage(self):
        """Check test suite coverage."""
        test_dirs = ["tests/unit/", "tests/integration/", "tests/characterization/"]
        test_coverage = {}

        for test_dir in test_dirs:
            if Path(test_dir).exists():
                test_files = list(Path(test_dir).glob("test_*.py"))
                test_coverage[test_dir] = len(test_files)
            else:
                test_coverage[test_dir] = 0

        total_tests = sum(test_coverage.values())

        if total_tests < 10:
            self.results["warnings"].append(f"Low test coverage: only {total_tests} test files")

        self.results["checks"]["test_coverage"] = {
            "status": "PASSED" if total_tests >= 10 else "WARNING",
            "test_files_by_type": test_coverage,
            "total_test_files": total_tests
        }

    def check_performance_configuration(self):
        """Check performance optimization configurations."""
        perf_features = []

        # Check for performance monitoring
        if Path("benchmarks/").exists():
            perf_features.append("Benchmarking suite available")

        # Check for memory optimization
        if Path("agents/memory_optimization/").exists():
            perf_features.append("Memory optimization modules present")

        # Check for connection pooling
        if Path("websocket/connection_pool.py").exists():
            perf_features.append("Connection pooling implemented")

        self.results["checks"]["performance_configuration"] = {
            "status": "PASSED" if len(perf_features) >= 2 else "WARNING",
            "performance_features": perf_features
        }

    def run_syntax_validation(self):
        """Run Python syntax validation on critical modules."""
        critical_modules = [
            "main.py",
            "api/main.py",
            "agents/__init__.py"
        ]

        syntax_errors = []

        for module in critical_modules:
            if Path(module).exists():
                try:
                    with open(module, 'r') as f:
                        compile(f.read(), module, 'exec')
                except SyntaxError as e:
                    syntax_errors.append(f"{module}: {str(e)}")

        if syntax_errors:
            self.results["blockers"].extend(syntax_errors)

        self.results["checks"]["syntax_validation"] = {
            "status": "PASSED" if not syntax_errors else "FAILED",
            "modules_checked": len(critical_modules),
            "syntax_errors": syntax_errors
        }

    def generate_deployment_recommendations(self):
        """Generate deployment recommendations based on findings."""
        recommendations = []

        # Security recommendations
        if self.results["checks"]["security_configuration"]["security_score"] < 75:
            recommendations.append("SECURITY: Implement additional security middleware")

        # Performance recommendations
        if len(self.results["checks"]["performance_configuration"]["performance_features"]) < 2:
            recommendations.append("PERFORMANCE: Add more performance optimization features")

        # Observability recommendations
        if not self.results["checks"]["observability_setup"]["observability_features"]:
            recommendations.append("MONITORING: Set up comprehensive monitoring stack")

        # General recommendations
        recommendations.append("DEPLOYMENT: Use production Docker configuration with health checks")
        recommendations.append("SCALING: Implement horizontal scaling with load balancer")
        recommendations.append("BACKUP: Set up automated database backup strategy")

        self.results["recommendations"] = recommendations

    def determine_overall_status(self):
        """Determine overall production readiness status."""
        failed_checks = [k for k, v in self.results["checks"].items() if v["status"] == "FAILED"]
        warning_checks = [k for k, v in self.results["checks"].items() if v["status"] == "WARNING"]

        if self.results["blockers"] or failed_checks:
            self.results["overall_status"] = "BLOCKED"
        elif len(warning_checks) > 2:
            self.results["overall_status"] = "CONDITIONAL"
        else:
            self.results["overall_status"] = "READY"

    def run_validation(self):
        """Run complete validation suite."""
        print("🚀 PIPELINE-ARCHITECT: Nemesis Production Readiness Validation")
        print("=" * 60)

        # Run all checks
        self.check_infrastructure_files()
        self.check_security_configuration()
        self.check_observability_setup()
        self.check_api_endpoints()
        self.check_database_migrations()
        self.check_test_coverage()
        self.check_performance_configuration()
        self.run_syntax_validation()

        # Generate recommendations and determine status
        self.generate_deployment_recommendations()
        self.determine_overall_status()

        # Output results
        print("\\n📊 VALIDATION RESULTS:")
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"Checks Completed: {len(self.results['checks'])}")
        print(f"Blockers: {len(self.results['blockers'])}")
        print(f"Warnings: {len(self.results['warnings'])}")

        if self.results["blockers"]:
            print("\\n❌ BLOCKERS:")
            for blocker in self.results["blockers"]:
                print(f"  - {blocker}")

        if self.results["warnings"]:
            print("\\n⚠️ WARNINGS:")
            for warning in self.results["warnings"]:
                print(f"  - {warning}")

        print("\\n✅ CHECK RESULTS:")
        for check, result in self.results["checks"].items():
            print(f"  - {check}: {result['status']}")

        return self.results

def main():
    """Main validation execution."""
    validator = ProductionReadinessValidator()
    results = validator.run_validation()

    # Save results to file
    with open("production_readiness_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\\n📄 Detailed results saved to: production_readiness_validation.json")

    # Exit with appropriate code
    if results["overall_status"] == "BLOCKED":
        print("\\n🚫 PRODUCTION DEPLOYMENT BLOCKED - Critical issues must be resolved")
        sys.exit(1)
    elif results["overall_status"] == "CONDITIONAL":
        print("\\n⚠️ CONDITIONAL APPROVAL - Address warnings before production deployment")
        sys.exit(0)
    else:
        print("\\n🎉 PRODUCTION READY - All critical validations passed")
        sys.exit(0)

if __name__ == "__main__":
    main()
