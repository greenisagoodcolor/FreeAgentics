#!/usr/bin/env python3
"""
Infrastructure Production Readiness Validation Script
Task 21: Validate Production Environment Configuration

This script validates production infrastructure and configuration without requiring
the application to be running.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InfrastructureValidator:
    """Production infrastructure validation"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "environment": "production",
            "validation_results": {},
            "summary": {},
        }
        self.critical_failures = []
        self.warnings = []
        self.passes = []

    def log_result(
        self, category: str, test_name: str, result: Dict[str, Any]
    ):
        """Log test result"""
        if category not in self.results["validation_results"]:
            self.results["validation_results"][category] = {}
        self.results["validation_results"][category][test_name] = result

        if result.get("status") == "CRITICAL":
            self.critical_failures.append(f"{category}: {test_name}")
            logger.error(
                f"CRITICAL: {category} - {test_name}: {result.get('message', 'Unknown error')}"
            )
        elif result.get("status") == "WARNING":
            self.warnings.append(f"{category}: {test_name}")
            logger.warning(
                f"WARNING: {category} - {test_name}: {result.get('message', 'Minor issue')}"
            )
        elif result.get("status") == "PASS":
            self.passes.append(f"{category}: {test_name}")
            logger.info(f"PASS: {category} - {test_name}")

    def validate_environment_configuration(self):
        """Validate environment configuration"""
        logger.info("Validating environment configuration...")

        # Check .env.production file
        env_file = ".env.production"
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                content = f.read()

            # Check for required variables
            required_vars = [
                "DATABASE_URL",
                "REDIS_URL",
                "SECRET_KEY",
                "JWT_SECRET",
                "ALLOWED_HOSTS",
                "MONITORING_ENABLED",
                "BACKUP_RETENTION_DAYS",
            ]

            missing_vars = []
            for var in required_vars:
                if f"{var}=" not in content:
                    missing_vars.append(var)

            self.log_result(
                "environment",
                "required_variables",
                {
                    "status": "PASS" if not missing_vars else "CRITICAL",
                    "missing_variables": missing_vars,
                    "message": (
                        f"Missing environment variables: {missing_vars}"
                        if missing_vars
                        else "All required variables present"
                    ),
                },
            )

            # Check for development values
            dev_patterns = [
                "localhost:5432",
                "password123",
                "your_",
                "dev_secret",
            ]
            dev_issues = [
                pattern for pattern in dev_patterns if pattern in content
            ]

            self.log_result(
                "environment",
                "production_values",
                {
                    "status": "WARNING" if dev_issues else "PASS",
                    "dev_issues": dev_issues,
                    "message": (
                        f"Development values found: {dev_issues}"
                        if dev_issues
                        else "No development values detected"
                    ),
                },
            )
        else:
            self.log_result(
                "environment",
                "env_file",
                {
                    "status": "CRITICAL",
                    "message": ".env.production file not found",
                },
            )

    def validate_docker_configuration(self):
        """Validate Docker configuration"""
        logger.info("Validating Docker configuration...")

        # Check Docker Compose production file
        compose_file = "docker-compose.production.yml"
        if os.path.exists(compose_file):
            with open(compose_file, "r") as f:
                try:
                    compose_config = yaml.safe_load(f)

                    # Check for required services
                    services = compose_config.get("services", {})
                    required_services = [
                        "postgres",
                        "redis",
                        "backend",
                        "nginx",
                    ]
                    missing_services = [
                        svc for svc in required_services if svc not in services
                    ]

                    self.log_result(
                        "docker",
                        "required_services",
                        {
                            "status": "PASS"
                            if not missing_services
                            else "CRITICAL",
                            "missing_services": missing_services,
                            "message": (
                                f"Missing services: {missing_services}"
                                if missing_services
                                else "All required services configured"
                            ),
                        },
                    )

                    # Check for security configurations
                    security_features = []
                    for service_name, service_config in services.items():
                        if service_config.get("read_only"):
                            security_features.append(
                                f"{service_name}: read-only"
                            )
                        if service_config.get("user"):
                            security_features.append(
                                f"{service_name}: non-root user"
                            )

                    self.log_result(
                        "docker",
                        "security_features",
                        {
                            "status": "PASS"
                            if security_features
                            else "WARNING",
                            "security_features": security_features,
                            "message": f"Security features: {len(security_features)} configured",
                        },
                    )

                except yaml.YAMLError as e:
                    self.log_result(
                        "docker",
                        "compose_syntax",
                        {
                            "status": "CRITICAL",
                            "message": f"Docker Compose syntax error: {e}",
                        },
                    )
        else:
            self.log_result(
                "docker",
                "compose_file",
                {
                    "status": "CRITICAL",
                    "message": "docker-compose.production.yml not found",
                },
            )

        # Check Dockerfile.production
        dockerfile = "Dockerfile.production"
        if os.path.exists(dockerfile):
            with open(dockerfile, "r") as f:
                content = f.read()

            security_checks = {
                "non_root_user": "USER " in content,
                "multi_stage_build": "FROM " in content
                and content.count("FROM") > 1,
                "health_check": "HEALTHCHECK" in content,
                "minimal_base": "slim" in content or "alpine" in content,
            }

            passed_checks = sum(security_checks.values())

            self.log_result(
                "docker",
                "dockerfile_security",
                {
                    "status": "PASS" if passed_checks >= 3 else "WARNING",
                    "security_checks": security_checks,
                    "passed_checks": passed_checks,
                    "message": f"Security checks passed: {passed_checks}/4",
                },
            )
        else:
            self.log_result(
                "docker",
                "dockerfile",
                {
                    "status": "CRITICAL",
                    "message": "Dockerfile.production not found",
                },
            )

    def validate_ssl_configuration(self):
        """Validate SSL/TLS configuration"""
        logger.info("Validating SSL/TLS configuration...")

        # Check SSL certificates
        ssl_files = [
            "nginx/ssl/cert.pem",
            "nginx/ssl/key.pem",
            "nginx/dhparam.pem",
        ]
        missing_files = [f for f in ssl_files if not os.path.exists(f)]

        self.log_result(
            "ssl",
            "certificate_files",
            {
                "status": "PASS" if not missing_files else "CRITICAL",
                "missing_files": missing_files,
                "message": (
                    f"Missing SSL files: {missing_files}"
                    if missing_files
                    else "All SSL files present"
                ),
            },
        )

        # Check nginx SSL configuration
        nginx_conf = "nginx/nginx.conf"
        if os.path.exists(nginx_conf):
            with open(nginx_conf, "r") as f:
                content = f.read()

            ssl_features = {
                "tls_modern": "TLSv1.2" in content and "TLSv1.3" in content,
                "hsts": "Strict-Transport-Security" in content,
                "strong_ciphers": "ECDHE" in content,
                "ssl_stapling": "ssl_stapling" in content,
            }

            passed_features = sum(ssl_features.values())

            self.log_result(
                "ssl",
                "nginx_configuration",
                {
                    "status": "PASS" if passed_features >= 3 else "WARNING",
                    "ssl_features": ssl_features,
                    "passed_features": passed_features,
                    "message": f"SSL features configured: {passed_features}/4",
                },
            )
        else:
            self.log_result(
                "ssl",
                "nginx_config",
                {"status": "CRITICAL", "message": "nginx.conf not found"},
            )

    def validate_security_configuration(self):
        """Validate security configuration"""
        logger.info("Validating security configuration...")

        # Check security modules
        security_modules = [
            "auth/security_headers.py",
            "auth/security_implementation.py",
            "auth/rbac_security_enhancements.py",
            "api/middleware/security_headers.py",
            "api/middleware/rate_limiter.py",
        ]

        existing_modules = [m for m in security_modules if os.path.exists(m)]

        self.log_result(
            "security",
            "security_modules",
            {
                "status": "PASS" if len(existing_modules) >= 4 else "WARNING",
                "existing_modules": existing_modules,
                "total_modules": len(security_modules),
                "message": f"Security modules: {len(existing_modules)}/{len(security_modules)} present",
            },
        )

        # Check JWT keys
        jwt_keys = ["auth/keys/jwt_private.pem", "auth/keys/jwt_public.pem"]
        missing_keys = [k for k in jwt_keys if not os.path.exists(k)]

        self.log_result(
            "security",
            "jwt_keys",
            {
                "status": "PASS" if not missing_keys else "CRITICAL",
                "missing_keys": missing_keys,
                "message": (
                    f"Missing JWT keys: {missing_keys}"
                    if missing_keys
                    else "JWT keys present"
                ),
            },
        )

        # Check security tests
        security_tests = Path("tests/security")
        if security_tests.exists():
            test_files = list(security_tests.glob("test_*.py"))

            self.log_result(
                "security",
                "security_tests",
                {
                    "status": "PASS" if len(test_files) >= 10 else "WARNING",
                    "test_count": len(test_files),
                    "message": f"Security tests: {len(test_files)} found",
                },
            )
        else:
            self.log_result(
                "security",
                "security_tests",
                {
                    "status": "WARNING",
                    "message": "Security tests directory not found",
                },
            )

    def validate_monitoring_configuration(self):
        """Validate monitoring configuration"""
        logger.info("Validating monitoring configuration...")

        # Check Prometheus configuration
        prometheus_configs = [
            "monitoring/prometheus.yml",
            "monitoring/prometheus-production.yml",
        ]

        prometheus_config = next(
            (c for c in prometheus_configs if os.path.exists(c)), None
        )

        if prometheus_config:
            with open(prometheus_config, "r") as f:
                try:
                    config = yaml.safe_load(f)

                    scrape_configs = config.get("scrape_configs", []) or []
                    rule_files = config.get("rule_files", []) or []

                    self.log_result(
                        "monitoring",
                        "prometheus_config",
                        {
                            "status": "PASS"
                            if scrape_configs and rule_files
                            else "WARNING",
                            "scrape_configs": len(scrape_configs),
                            "rule_files": len(rule_files),
                            "message": f"Prometheus: {len(scrape_configs)} scrape configs, {len(rule_files)} rule files",
                        },
                    )

                except yaml.YAMLError as e:
                    self.log_result(
                        "monitoring",
                        "prometheus_config",
                        {
                            "status": "CRITICAL",
                            "message": f"Prometheus config error: {e}",
                        },
                    )
        else:
            self.log_result(
                "monitoring",
                "prometheus_config",
                {
                    "status": "CRITICAL",
                    "message": "Prometheus configuration not found",
                },
            )

        # Check Grafana dashboards
        dashboard_dir = Path("monitoring/grafana/dashboards")
        if dashboard_dir.exists():
            dashboards = list(dashboard_dir.glob("*.json"))

            self.log_result(
                "monitoring",
                "grafana_dashboards",
                {
                    "status": "PASS" if len(dashboards) >= 5 else "WARNING",
                    "dashboard_count": len(dashboards),
                    "message": f"Grafana dashboards: {len(dashboards)} found",
                },
            )
        else:
            self.log_result(
                "monitoring",
                "grafana_dashboards",
                {
                    "status": "WARNING",
                    "message": "Grafana dashboards directory not found",
                },
            )

        # Check alert rules
        alert_rules = [
            "monitoring/rules/alerts.yml",
            "monitoring/prometheus-alerts.yml",
        ]

        existing_rules = [r for r in alert_rules if os.path.exists(r)]

        self.log_result(
            "monitoring",
            "alert_rules",
            {
                "status": "PASS" if existing_rules else "WARNING",
                "existing_rules": existing_rules,
                "message": f"Alert rules: {len(existing_rules)} files found",
            },
        )

    def validate_backup_configuration(self):
        """Validate backup configuration"""
        logger.info("Validating backup configuration...")

        # Check backup scripts
        backup_scripts = [
            "scripts/database-backup.sh",
            "scripts/backup/full-backup.sh",
        ]

        existing_scripts = [s for s in backup_scripts if os.path.exists(s)]

        self.log_result(
            "backup",
            "backup_scripts",
            {
                "status": "PASS" if existing_scripts else "CRITICAL",
                "existing_scripts": existing_scripts,
                "message": f"Backup scripts: {len(existing_scripts)} found",
            },
        )

        # Check backup documentation
        backup_docs = [
            "docs/operations/BACKUP_RECOVERY.md",
            "docs/runbooks/BACKUP_RECOVERY.md",
        ]

        existing_docs = [d for d in backup_docs if os.path.exists(d)]

        self.log_result(
            "backup",
            "backup_documentation",
            {
                "status": "PASS" if existing_docs else "WARNING",
                "existing_docs": existing_docs,
                "message": f"Backup documentation: {len(existing_docs)} found",
            },
        )

    def validate_disaster_recovery(self):
        """Validate disaster recovery configuration"""
        logger.info("Validating disaster recovery configuration...")

        # Check disaster recovery scripts
        dr_scripts = [
            "scripts/rollback.sh",
            "scripts/deployment/rollback.sh",
            "scripts/disaster-recovery.sh",
        ]

        existing_scripts = [s for s in dr_scripts if os.path.exists(s)]

        self.log_result(
            "disaster_recovery",
            "recovery_scripts",
            {
                "status": "PASS" if existing_scripts else "WARNING",
                "existing_scripts": existing_scripts,
                "message": f"Recovery scripts: {len(existing_scripts)} found",
            },
        )

        # Check disaster recovery documentation
        dr_docs = [
            "docs/runbooks/DISASTER_RECOVERY_RUNBOOK.md",
            "docs/runbooks/EMERGENCY_PROCEDURES.md",
            "docs/runbooks/INCIDENT_RESPONSE.md",
        ]

        existing_docs = [d for d in dr_docs if os.path.exists(d)]

        self.log_result(
            "disaster_recovery",
            "recovery_documentation",
            {
                "status": "PASS" if len(existing_docs) >= 2 else "WARNING",
                "existing_docs": existing_docs,
                "message": f"DR documentation: {len(existing_docs)} found",
            },
        )

    def validate_deployment_scripts(self):
        """Validate deployment scripts"""
        logger.info("Validating deployment scripts...")

        # Check deployment scripts
        deployment_scripts = [
            "deploy-production.sh",
            "scripts/deployment/deploy-production.sh",
        ]

        existing_scripts = [s for s in deployment_scripts if os.path.exists(s)]

        if existing_scripts:
            script_path = existing_scripts[0]
            with open(script_path, "r") as f:
                content = f.read()

            # Check for deployment features
            deployment_features = {
                "health_checks": "health" in content.lower(),
                "rollback_capability": "rollback" in content.lower(),
                "zero_downtime": "zero" in content.lower()
                or "blue" in content.lower(),
                "database_migration": "migrate" in content.lower()
                or "alembic" in content.lower(),
            }

            passed_features = sum(deployment_features.values())

            self.log_result(
                "deployment",
                "deployment_features",
                {
                    "status": "PASS" if passed_features >= 2 else "WARNING",
                    "deployment_features": deployment_features,
                    "passed_features": passed_features,
                    "message": f"Deployment features: {passed_features}/4 present",
                },
            )
        else:
            self.log_result(
                "deployment",
                "deployment_scripts",
                {
                    "status": "CRITICAL",
                    "message": "No deployment scripts found",
                },
            )

    def validate_test_coverage(self):
        """Validate test coverage"""
        logger.info("Validating test coverage...")

        # Check test directories
        test_dirs = [
            "tests/unit",
            "tests/integration",
            "tests/security",
            "tests/performance",
        ]
        existing_dirs = [d for d in test_dirs if os.path.exists(d)]

        self.log_result(
            "testing",
            "test_directories",
            {
                "status": "PASS" if len(existing_dirs) >= 3 else "WARNING",
                "existing_dirs": existing_dirs,
                "message": f"Test directories: {len(existing_dirs)}/4 present",
            },
        )

        # Count test files
        total_tests = 0
        for test_dir in existing_dirs:
            test_files = list(Path(test_dir).glob("test_*.py"))
            total_tests += len(test_files)

        self.log_result(
            "testing",
            "test_coverage",
            {
                "status": "PASS" if total_tests >= 50 else "WARNING",
                "total_tests": total_tests,
                "message": f"Test files: {total_tests} found",
            },
        )

    def run_validation(self):
        """Run all validation checks"""
        logger.info("Starting infrastructure validation...")

        # Run all validation checks
        self.validate_environment_configuration()
        self.validate_docker_configuration()
        self.validate_ssl_configuration()
        self.validate_security_configuration()
        self.validate_monitoring_configuration()
        self.validate_backup_configuration()
        self.validate_disaster_recovery()
        self.validate_deployment_scripts()
        self.validate_test_coverage()

        # Calculate summary
        total_tests = (
            len(self.passes) + len(self.warnings) + len(self.critical_failures)
        )
        pass_rate = (
            (len(self.passes) / total_tests) * 100 if total_tests > 0 else 0
        )

        self.results["summary"] = {
            "total_tests": total_tests,
            "passed": len(self.passes),
            "warnings": len(self.warnings),
            "critical_failures": len(self.critical_failures),
            "pass_rate": pass_rate,
            "production_ready": len(self.critical_failures) == 0,
        }

        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"infrastructure_validation_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Generate markdown report
        markdown_file = f"infrastructure_validation_report_{timestamp}.md"
        self.generate_markdown_report(markdown_file)

        # Print summary
        print("\n" + "=" * 60)
        print("  INFRASTRUCTURE VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.results['summary']['total_tests']}")
        print(f"Passed: {self.results['summary']['passed']}")
        print(f"Warnings: {self.results['summary']['warnings']}")
        print(
            f"Critical Failures: {self.results['summary']['critical_failures']}"
        )
        print(f"Pass Rate: {self.results['summary']['pass_rate']:.1f}%")
        print(
            f"Production Ready: {'YES' if self.results['summary']['production_ready'] else 'NO'}"
        )
        print("=" * 60)

        if self.results["summary"]["production_ready"]:
            print("🎉 INFRASTRUCTURE READY FOR PRODUCTION! 🚀")
        else:
            print("❌ INFRASTRUCTURE NOT READY - Fix critical issues:")
            for failure in self.critical_failures:
                print(f"  - {failure}")

        print(f"\nReports generated: {report_file}, {markdown_file}")

        return self.results["summary"]["production_ready"]

    def generate_markdown_report(self, filename: str):
        """Generate markdown report"""
        with open(filename, "w") as f:
            f.write("# FreeAgentics Infrastructure Validation Report\n\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"**Environment:** {self.results['environment']}\n\n")

            # Summary
            summary = self.results["summary"]
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Tests:** {summary['total_tests']}\n")
            f.write(f"- **Passed:** {summary['passed']}\n")
            f.write(f"- **Warnings:** {summary['warnings']}\n")
            f.write(
                f"- **Critical Failures:** {summary['critical_failures']}\n"
            )
            f.write(f"- **Pass Rate:** {summary['pass_rate']:.1f}%\n")
            f.write(
                f"- **Production Ready:** {'✅ YES' if summary['production_ready'] else '❌ NO'}\n\n"
            )

            # Critical failures
            if self.critical_failures:
                f.write("## 🚨 Critical Issues (Must Fix)\n\n")
                for failure in self.critical_failures:
                    f.write(f"- {failure}\n")
                f.write("\n")

            # Warnings
            if self.warnings:
                f.write("## ⚠️ Warnings (Recommended Fixes)\n\n")
                for warning in self.warnings:
                    f.write(f"- {warning}\n")
                f.write("\n")

            # Detailed results
            f.write("## Detailed Validation Results\n\n")
            for category, tests in self.results["validation_results"].items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                for test_name, result in tests.items():
                    status_icon = {
                        "PASS": "✅",
                        "WARNING": "⚠️",
                        "CRITICAL": "❌",
                    }.get(result.get("status", ""), "ℹ️")
                    f.write(
                        f"- {status_icon} **{test_name}**: {result.get('message', 'No message')}\n"
                    )
                f.write("\n")


def main():
    """Main function"""
    validator = InfrastructureValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
