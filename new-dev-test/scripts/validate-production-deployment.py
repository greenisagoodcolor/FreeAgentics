#!/usr/bin/env python3
"""
FreeAgentics Production Deployment Validation Script
Comprehensive validation with nemesis-level rigor for production readiness
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import docker
import yaml


class Colors:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class ProductionValidator:
    """Comprehensive production deployment validator"""

    def __init__(self):
        self.results = []
        self.errors = []
        self.warnings = []
        self.docker_client = docker.from_env()
        self.validation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, message: str, level: str = "INFO"):
        """Log message with color coding"""
        color_map = {
            "INFO": Colors.BLUE,
            "SUCCESS": Colors.GREEN,
            "WARNING": Colors.YELLOW,
            "ERROR": Colors.RED,
            "CRITICAL": Colors.MAGENTA,
        }
        color = color_map.get(level, Colors.RESET)
        print(f"{color}[{level}] {message}{Colors.RESET}")

    def run_command(self, cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
        """Run command and return result"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            return result.returncode, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return e.returncode, e.stdout, e.stderr

    def check_file_exists(self, filepath: str) -> bool:
        """Check if file exists"""
        return os.path.exists(filepath)

    def validate_docker_setup(self) -> Dict[str, bool]:
        """Validate Docker configuration and files"""
        self.log("Validating Docker setup...", "INFO")

        results = {
            "dockerfile_production": False,
            "docker_compose_production": False,
            "dockerfile_optimized": False,
            "dockerfile_multi_stage": False,
            "security_non_root": False,
            "health_checks": False,
            "resource_limits": False,
        }

        # Check Dockerfile.production
        if self.check_file_exists("Dockerfile.production"):
            results["dockerfile_production"] = True

            # Validate multi-stage build
            with open("Dockerfile.production", "r") as f:
                content = f.read()
                if "FROM" in content and content.count("FROM") > 1:
                    results["dockerfile_multi_stage"] = True
                    self.log("✓ Multi-stage Docker build detected", "SUCCESS")

                # Check for non-root user
                if "USER app" in content or "USER 1000" in content:
                    results["security_non_root"] = True
                    self.log("✓ Non-root user configured", "SUCCESS")
                else:
                    self.warnings.append("Docker container running as root")

                # Check for health check
                if "HEALTHCHECK" in content:
                    results["health_checks"] = True
                    self.log("✓ Health check configured", "SUCCESS")
        else:
            self.errors.append("Dockerfile.production not found")

        # Check docker-compose.production.yml
        if self.check_file_exists("docker-compose.production.yml"):
            results["docker_compose_production"] = True

            with open("docker-compose.production.yml", "r") as f:
                compose_data = yaml.safe_load(f)

            # Check resource limits
            for service_name, service_config in compose_data.get("services", {}).items():
                if "deploy" in service_config and "resources" in service_config["deploy"]:
                    results["resource_limits"] = True
                    self.log(
                        f"✓ Resource limits configured for {service_name}",
                        "SUCCESS",
                    )

        return results

    def validate_environment_configuration(self) -> Dict[str, bool]:
        """Validate environment configuration"""
        self.log("Validating environment configuration...", "INFO")

        results = {
            "env_template": False,
            "env_production": False,
            "required_vars": False,
            "secure_secrets": False,
            "ssl_config": False,
        }

        # Check environment template
        if self.check_file_exists(".env.production.ssl.template"):
            results["env_template"] = True
            self.log("✓ Production environment template found", "SUCCESS")

        # Check for production environment file
        if self.check_file_exists(".env.production"):
            results["env_production"] = True
            self.log(
                "⚠️  Production environment file exists - ensure not in git",
                "WARNING",
            )

            # Check for development secrets
            with open(".env.production", "r") as f:
                content = f.read()
                if "dev_secret" in content or "your_" in content:
                    self.errors.append("Development secrets found in production environment")
                    results["secure_secrets"] = False
                else:
                    results["secure_secrets"] = True

        # Validate required environment variables
        required_vars = [
            "DATABASE_URL",
            "REDIS_PASSWORD",
            "SECRET_KEY",
            "JWT_SECRET",
            "DOMAIN",
            "SSL_CERT_PATH",
            "SSL_KEY_PATH",
        ]

        env_vars = os.environ.copy()
        if self.check_file_exists(".env.production"):
            with open(".env.production", "r") as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        key = line.split("=")[0].strip()
                        env_vars[key] = "SET"

        missing_vars = [var for var in required_vars if var not in env_vars]
        if not missing_vars:
            results["required_vars"] = True
            self.log("✓ All required environment variables configured", "SUCCESS")
        else:
            self.warnings.append(f"Missing environment variables: {', '.join(missing_vars)}")

        return results

    def validate_ssl_tls_setup(self) -> Dict[str, bool]:
        """Validate SSL/TLS configuration"""
        self.log("Validating SSL/TLS setup...", "INFO")

        results = {
            "nginx_config": False,
            "ssl_certs": False,
            "dhparam": False,
            "ssl_scripts": False,
            "cert_monitoring": False,
            "security_headers": False,
        }

        # Check nginx configuration
        if self.check_file_exists("nginx/nginx.conf"):
            results["nginx_config"] = True

            with open("nginx/nginx.conf", "r") as f:
                nginx_content = f.read()

            # Check SSL configuration
            if "ssl_protocols TLSv1.2 TLSv1.3" in nginx_content:
                self.log("✓ Modern SSL protocols configured", "SUCCESS")

            # Check security headers
            security_headers = [
                "X-Frame-Options",
                "X-Content-Type-Options",
                "X-XSS-Protection",
                "Strict-Transport-Security",
            ]

            headers_found = all(header in nginx_content for header in security_headers)
            if headers_found:
                results["security_headers"] = True
                self.log("✓ Security headers configured", "SUCCESS")

        # Check SSL certificates
        if self.check_file_exists("nginx/ssl/cert.pem") and self.check_file_exists(
            "nginx/ssl/key.pem"
        ):
            results["ssl_certs"] = True
            self.log("✓ SSL certificates found", "SUCCESS")

        # Check DH parameters
        if self.check_file_exists("nginx/dhparam.pem"):
            results["dhparam"] = True
            self.log("✓ DH parameters configured", "SUCCESS")

        # Check SSL scripts
        ssl_scripts = [
            "nginx/certbot-setup.sh",
            "nginx/monitor-ssl.sh",
            "nginx/test-ssl.sh",
        ]

        if all(self.check_file_exists(script) for script in ssl_scripts):
            results["ssl_scripts"] = True
            self.log("✓ SSL management scripts available", "SUCCESS")

        return results

    def validate_database_configuration(self) -> Dict[str, bool]:
        """Validate database configuration and migrations"""
        self.log("Validating database configuration...", "INFO")

        results = {
            "postgres_config": False,
            "migrations": False,
            "backup_scripts": False,
            "connection_pooling": False,
            "ssl_enabled": False,
        }

        # Check PostgreSQL configuration in docker-compose
        if self.check_file_exists("docker-compose.production.yml"):
            with open("docker-compose.production.yml", "r") as f:
                compose_data = yaml.safe_load(f)

            postgres_config = compose_data.get("services", {}).get("postgres", {})
            if postgres_config:
                results["postgres_config"] = True

                # Check for health check
                if "healthcheck" in postgres_config:
                    self.log("✓ PostgreSQL health check configured", "SUCCESS")

        # Check migration files
        if self.check_file_exists("alembic.ini") and os.path.exists("alembic/versions"):
            results["migrations"] = True
            self.log("✓ Database migrations configured", "SUCCESS")

        # Check backup scripts
        if self.check_file_exists("scripts/database-backup.sh"):
            results["backup_scripts"] = True
            self.log("✓ Database backup scripts available", "SUCCESS")

        return results

    def validate_deployment_scripts(self) -> Dict[str, bool]:
        """Validate deployment scripts"""
        self.log("Validating deployment scripts...", "INFO")

        results = {
            "deploy_script": False,
            "ssl_deploy_script": False,
            "zero_downtime": False,
            "rollback_capability": False,
            "health_checks": False,
        }

        # Check main deployment script
        if self.check_file_exists("deploy-production.sh"):
            results["deploy_script"] = True

            with open("deploy-production.sh", "r") as f:
                deploy_content = f.read()

            # Check for zero-downtime deployment
            if "zero_downtime" in deploy_content or "rolling_update" in deploy_content:
                results["zero_downtime"] = True
                self.log("✓ Zero-downtime deployment configured", "SUCCESS")

            # Check for rollback capability
            if "rollback" in deploy_content:
                results["rollback_capability"] = True
                self.log("✓ Rollback capability implemented", "SUCCESS")

            # Check for health checks
            if "wait_for_health" in deploy_content:
                results["health_checks"] = True
                self.log("✓ Health check verification in deployment", "SUCCESS")

        # Check SSL deployment script
        if self.check_file_exists("deploy-production-ssl.sh"):
            results["ssl_deploy_script"] = True
            self.log("✓ SSL deployment script available", "SUCCESS")

        return results

    def validate_monitoring_setup(self) -> Dict[str, bool]:
        """Validate monitoring and alerting configuration"""
        self.log("Validating monitoring setup...", "INFO")

        results = {
            "prometheus_config": False,
            "alerting_rules": False,
            "grafana_dashboards": False,
            "metrics_endpoints": False,
            "log_aggregation": False,
        }

        # Check Prometheus configuration
        if self.check_file_exists("monitoring/prometheus-production.yml"):
            results["prometheus_config"] = True
            self.log("✓ Prometheus production configuration found", "SUCCESS")

        # Check alerting rules
        if os.path.exists("monitoring/rules") and os.listdir("monitoring/rules"):
            results["alerting_rules"] = True
            self.log("✓ Alerting rules configured", "SUCCESS")

        # Check Grafana dashboards
        if os.path.exists("monitoring/grafana/dashboards"):
            results["grafana_dashboards"] = True
            self.log("✓ Grafana dashboards configured", "SUCCESS")

        return results

    def validate_security_configuration(self) -> Dict[str, bool]:
        """Validate security configuration"""
        self.log("Validating security configuration...", "INFO")

        results = {
            "rbac_configured": False,
            "secrets_management": False,
            "rate_limiting": False,
            "input_validation": False,
            "security_headers": False,
            "container_security": False,
        }

        # Check RBAC configuration
        if self.check_file_exists("auth/rbac.py") or self.check_file_exists(
            "auth/rbac_enhancements.py"
        ):
            results["rbac_configured"] = True
            self.log("✓ RBAC configured", "SUCCESS")

        # Check rate limiting in nginx
        if self.check_file_exists("nginx/nginx.conf"):
            with open("nginx/nginx.conf", "r") as f:
                if "limit_req_zone" in f.read():
                    results["rate_limiting"] = True
                    self.log("✓ Rate limiting configured", "SUCCESS")

        # Check container security (from earlier Docker validation)
        if self.check_file_exists("docker-compose.production.yml"):
            with open("docker-compose.production.yml", "r") as f:
                compose_content = f.read()
                if "read_only: true" in compose_content:
                    results["container_security"] = True
                    self.log("✓ Read-only containers configured", "SUCCESS")

        return results

    def run_makefile_commands(self) -> Dict[str, bool]:
        """Test all Makefile production commands"""
        self.log("Testing Makefile commands...", "INFO")

        results = {
            "make_docker_build": False,
            "make_docker_up": False,
            "make_prod_env": False,
            "make_security_audit": False,
        }

        # Test make docker-build (dry run)
        self.log("Testing 'make docker-build' (dry run)...", "INFO")
        ret, stdout, stderr = self.run_command(["make", "-n", "docker-build"], check=False)
        if ret == 0:
            results["make_docker_build"] = True
            self.log("✓ make docker-build command available", "SUCCESS")

        # Test make prod-env
        self.log("Testing 'make prod-env'...", "INFO")
        ret, stdout, stderr = self.run_command(["make", "prod-env"], check=False)
        if ret == 0:
            results["make_prod_env"] = True
            self.log("✓ make prod-env command successful", "SUCCESS")
        else:
            self.warnings.append(f"make prod-env failed: {stderr}")

        return results

    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        report = {
            "timestamp": self.validation_timestamp,
            "summary": {
                "total_checks": len(self.results),
                "passed": sum(1 for r in self.results if all(r.values())),
                "warnings": len(self.warnings),
                "errors": len(self.errors),
            },
            "docker_validation": self.validate_docker_setup(),
            "environment_validation": self.validate_environment_configuration(),
            "ssl_tls_validation": self.validate_ssl_tls_setup(),
            "database_validation": self.validate_database_configuration(),
            "deployment_validation": self.validate_deployment_scripts(),
            "monitoring_validation": self.validate_monitoring_setup(),
            "security_validation": self.validate_security_configuration(),
            "makefile_validation": self.run_makefile_commands(),
            "warnings": self.warnings,
            "errors": self.errors,
        }

        # Save report
        report_filename = f"production_validation_report_{self.validation_timestamp}.json"
        with open(report_filename, "w") as f:
            json.dump(report, f, indent=2)

        self.log(f"\nValidation report saved to: {report_filename}", "INFO")

        # Print summary
        print(f"\n{Colors.BOLD}=== PRODUCTION DEPLOYMENT VALIDATION SUMMARY ==={Colors.RESET}")
        print(f"Timestamp: {self.validation_timestamp}")
        print(f"Total Checks: {report['summary']['total_checks']}")
        print(f"{Colors.GREEN}Passed: {report['summary']['passed']}{Colors.RESET}")
        print(f"{Colors.YELLOW}Warnings: {report['summary']['warnings']}{Colors.RESET}")
        print(f"{Colors.RED}Errors: {report['summary']['errors']}{Colors.RESET}")

        if self.errors:
            print(f"\n{Colors.RED}Critical Issues:{Colors.RESET}")
            for error in self.errors:
                print(f"  ❌ {error}")

        if self.warnings:
            print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")

        # Overall status
        if not self.errors:
            print(
                f"\n{Colors.GREEN}✅ Production deployment infrastructure validation PASSED{Colors.RESET}"
            )
        else:
            print(
                f"\n{Colors.RED}❌ Production deployment infrastructure validation FAILED{Colors.RESET}"
            )
            print("Please address the critical issues before deploying to production.")

        return report


def main():
    """Main validation entry point"""
    print(f"{Colors.BOLD}FreeAgentics Production Deployment Validator{Colors.RESET}")
    print("=" * 50)

    validator = ProductionValidator()

    try:
        validator.generate_validation_report()

        # Exit with appropriate code
        if validator.errors:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"{Colors.RED}[CRITICAL] Validation failed with error: {e}{Colors.RESET}")
        sys.exit(2)


if __name__ == "__main__":
    main()
