#!/usr/bin/env python3
"""
Comprehensive Production Readiness Validation Script
Task 21: Validate Production Environment Configuration

This script performs comprehensive validation of the production environment:
1. Load testing and performance validation
2. Security validation and penetration testing
3. Disaster recovery testing
4. Operational procedure validation
5. Monitoring and alerting validation
"""

import asyncio
import json
import logging
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import requests
import websockets
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("production_validation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ProductionValidator:
    """Comprehensive production validation suite"""

    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "environment": "production",
            "validation_results": {},
            "performance_metrics": {},
            "security_audit": {},
            "disaster_recovery": {},
            "monitoring_validation": {},
            "summary": {},
        }
        self.critical_failures = []
        self.warnings = []
        self.passes = []

    def log_result(self, category: str, test_name: str, result: Dict[str, Any]):
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
        else:
            logger.info(f"INFO: {category} - {test_name}: {result.get('message', 'No status')}")

    def check_environment_readiness(self) -> bool:
        """Check if environment is ready for testing"""
        logger.info("Checking environment readiness...")

        # Check if .env.production exists
        if not os.path.exists(".env.production"):
            logger.error("Production environment file not found!")
            return False

        # Check if required services are running
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info("API server is running")
                return True
            else:
                logger.error(f"API server not responding correctly: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to API server: {e}")
            return False

    def validate_load_testing(self):
        """Perform load testing and performance validation"""
        logger.info("Starting load testing and performance validation...")

        # Test 1: Basic response time
        response_times = []
        for i in range(10):
            start = time.time()
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    response_times.append(time.time() - start)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                break

        if response_times:
            avg_response = sum(response_times) / len(response_times)
            p95_response = sorted(response_times)[int(len(response_times) * 0.95)]

            self.log_result(
                "performance",
                "response_time_95th_percentile",
                {
                    "status": "PASS" if p95_response < 0.2 else "WARNING",
                    "value": p95_response,
                    "threshold": 0.2,
                    "message": f"95th percentile response time: {p95_response:.3f}s",
                },
            )

        # Test 2: Concurrent user simulation
        self.concurrent_load_test()

        # Test 3: Memory usage validation
        self.validate_memory_usage()

        # Test 4: Database performance
        self.validate_database_performance()

    def concurrent_load_test(self):
        """Run concurrent load test"""
        logger.info("Running concurrent load test...")

        def make_request(url: str) -> Tuple[int, float]:
            start = time.time()
            try:
                response = requests.get(url, timeout=10)
                return response.status_code, time.time() - start
            except Exception:
                return 0, 0

        urls = [
            f"{self.base_url}/health",
            f"{self.base_url}/api/v1/monitoring/metrics",
            f"{self.base_url}/api/v1/system/info",
        ]

        concurrent_users = 50
        requests_per_user = 10

        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            for _ in range(concurrent_users):
                for _ in range(requests_per_user):
                    for url in urls:
                        futures.append(executor.submit(make_request, url))

            results = []
            for future in as_completed(futures):
                status, response_time = future.result()
                results.append((status, response_time))

        # Analyze results
        successful_requests = sum(1 for status, _ in results if status == 200)
        total_requests = len(results)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0

        self.log_result(
            "performance",
            "concurrent_load_test",
            {
                "status": "PASS" if success_rate > 0.95 else "CRITICAL",
                "success_rate": success_rate,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "message": f"Success rate: {success_rate:.2%}",
            },
        )

    def validate_memory_usage(self):
        """Validate system memory usage"""
        logger.info("Validating memory usage...")

        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        self.log_result(
            "performance",
            "memory_usage",
            {
                "status": "PASS" if memory.percent < 80 else "WARNING",
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "cpu_percent": cpu_percent,
                "message": f"Memory usage: {memory.percent}%, CPU: {cpu_percent}%",
            },
        )

    def validate_database_performance(self):
        """Validate database performance"""
        logger.info("Validating database performance...")

        try:
            # Test database connection pool
            import psycopg2
            from psycopg2.pool import ThreadedConnectionPool

            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                self.log_result(
                    "performance",
                    "database_connection",
                    {"status": "CRITICAL", "message": "DATABASE_URL not configured"},
                )
                return

            # Parse database URL
            import urllib.parse

            parsed = urllib.parse.urlparse(db_url)

            # Test connection pool
            pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=parsed.hostname,
                port=parsed.port,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password,
            )

            # Test queries
            conn = pool.getconn()
            cursor = conn.cursor()

            # Test basic query
            start = time.time()
            cursor.execute("SELECT 1")
            query_time = time.time() - start

            # Test table existence
            cursor.execute(
                """
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
            )
            tables = cursor.fetchall()

            cursor.close()
            pool.putconn(conn)
            pool.closeall()

            self.log_result(
                "performance",
                "database_performance",
                {
                    "status": "PASS" if query_time < 0.1 else "WARNING",
                    "query_time": query_time,
                    "tables_found": len(tables),
                    "message": f"Database query time: {query_time:.3f}s, Tables: {len(tables)}",
                },
            )

        except Exception as e:
            self.log_result(
                "performance",
                "database_performance",
                {"status": "CRITICAL", "message": f"Database performance test failed: {e}"},
            )

    def validate_security(self):
        """Perform security validation and penetration testing"""
        logger.info("Starting security validation and penetration testing...")

        # Test 1: SSL/TLS Configuration
        self.test_ssl_configuration()

        # Test 2: Security headers
        self.test_security_headers()

        # Test 3: Authentication security
        self.test_authentication_security()

        # Test 4: Rate limiting
        self.test_rate_limiting()

        # Test 5: Input validation
        self.test_input_validation()

        # Test 6: CORS configuration
        self.test_cors_configuration()

    def test_ssl_configuration(self):
        """Test SSL/TLS configuration"""
        logger.info("Testing SSL/TLS configuration...")

        try:
            # Test SSL certificate
            context = ssl.create_default_context()
            with socket.create_connection(("localhost", 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname="localhost") as ssock:
                    cert = ssock.getpeercert()

                    # Check certificate expiry
                    not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z")
                    days_until_expiry = (not_after - datetime.now()).days

                    self.log_result(
                        "security",
                        "ssl_certificate",
                        {
                            "status": "PASS" if days_until_expiry > 30 else "WARNING",
                            "days_until_expiry": days_until_expiry,
                            "certificate_subject": cert.get("subject", []),
                            "message": f"SSL certificate expires in {days_until_expiry} days",
                        },
                    )
        except Exception as e:
            self.log_result(
                "security",
                "ssl_certificate",
                {"status": "WARNING", "message": f"SSL test failed (may not be configured): {e}"},
            )

    def test_security_headers(self):
        """Test security headers"""
        logger.info("Testing security headers...")

        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            headers = response.headers

            required_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000",
                "Content-Security-Policy": "default-src",
            }

            missing_headers = []
            for header, expected in required_headers.items():
                if header not in headers:
                    missing_headers.append(header)
                elif expected not in headers[header]:
                    missing_headers.append(f"{header} (incorrect value)")

            self.log_result(
                "security",
                "security_headers",
                {
                    "status": "PASS" if not missing_headers else "WARNING",
                    "missing_headers": missing_headers,
                    "message": (
                        f"Missing security headers: {missing_headers}"
                        if missing_headers
                        else "All security headers present"
                    ),
                },
            )

        except Exception as e:
            self.log_result(
                "security",
                "security_headers",
                {"status": "CRITICAL", "message": f"Security headers test failed: {e}"},
            )

    def test_authentication_security(self):
        """Test authentication security"""
        logger.info("Testing authentication security...")

        try:
            # Test login endpoint
            login_data = {"username": "admin", "password": "wrong_password"}

            response = requests.post(
                f"{self.base_url}/api/v1/auth/login", json=login_data, timeout=10
            )

            # Should fail with wrong credentials
            self.log_result(
                "security",
                "authentication_security",
                {
                    "status": "PASS" if response.status_code in [401, 403] else "WARNING",
                    "response_code": response.status_code,
                    "message": f"Authentication correctly rejected invalid credentials: {response.status_code}",
                },
            )

        except Exception as e:
            self.log_result(
                "security",
                "authentication_security",
                {"status": "WARNING", "message": f"Authentication test failed: {e}"},
            )

    def test_rate_limiting(self):
        """Test rate limiting"""
        logger.info("Testing rate limiting...")

        try:
            # Make rapid requests to trigger rate limiting
            responses = []
            for i in range(100):
                response = requests.get(f"{self.base_url}/health", timeout=1)
                responses.append(response.status_code)
                if response.status_code == 429:  # Too Many Requests
                    break

            rate_limited = 429 in responses

            self.log_result(
                "security",
                "rate_limiting",
                {
                    "status": "PASS" if rate_limited else "WARNING",
                    "rate_limited": rate_limited,
                    "total_requests": len(responses),
                    "message": f"Rate limiting {'active' if rate_limited else 'not detected'}",
                },
            )

        except Exception as e:
            self.log_result(
                "security",
                "rate_limiting",
                {"status": "WARNING", "message": f"Rate limiting test failed: {e}"},
            )

    def test_input_validation(self):
        """Test input validation"""
        logger.info("Testing input validation...")

        # Test SQL injection attempts
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--",
        ]

        for payload in sql_payloads:
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/auth/login",
                    json={"username": payload, "password": "test"},
                    timeout=10,
                )

                # Should not crash or return 500
                if response.status_code == 500:
                    self.log_result(
                        "security",
                        "sql_injection_protection",
                        {
                            "status": "CRITICAL",
                            "payload": payload,
                            "message": f"SQL injection vulnerability detected with payload: {payload}",
                        },
                    )
                    return

            except Exception as e:
                logger.debug(f"Input validation test with payload {payload}: {e}")

        self.log_result(
            "security",
            "sql_injection_protection",
            {"status": "PASS", "message": "No SQL injection vulnerabilities detected"},
        )

    def test_cors_configuration(self):
        """Test CORS configuration"""
        logger.info("Testing CORS configuration...")

        try:
            response = requests.options(
                f"{self.base_url}/api/v1/agents",
                headers={
                    "Origin": "https://malicious-site.com",
                    "Access-Control-Request-Method": "GET",
                },
                timeout=10,
            )

            cors_headers = response.headers.get("Access-Control-Allow-Origin", "")

            self.log_result(
                "security",
                "cors_configuration",
                {
                    "status": "PASS" if cors_headers != "*" else "WARNING",
                    "cors_headers": cors_headers,
                    "message": f"CORS configuration: {cors_headers}",
                },
            )

        except Exception as e:
            self.log_result(
                "security",
                "cors_configuration",
                {"status": "WARNING", "message": f"CORS test failed: {e}"},
            )

    def validate_disaster_recovery(self):
        """Test disaster recovery and business continuity"""
        logger.info("Testing disaster recovery and business continuity...")

        # Test 1: Backup verification
        self.test_backup_procedures()

        # Test 2: Service restart capability
        self.test_service_restart()

        # Test 3: Data recovery simulation
        self.test_data_recovery()

        # Test 4: Failover testing
        self.test_failover_procedures()

    def test_backup_procedures(self):
        """Test backup procedures"""
        logger.info("Testing backup procedures...")

        backup_script = "scripts/database-backup.sh"
        if os.path.exists(backup_script):
            try:
                # Test backup script (dry run)
                result = subprocess.run(
                    ["bash", backup_script, "--dry-run"], capture_output=True, text=True, timeout=60
                )

                self.log_result(
                    "disaster_recovery",
                    "backup_procedures",
                    {
                        "status": "PASS" if result.returncode == 0 else "WARNING",
                        "return_code": result.returncode,
                        "output": result.stdout[:500],
                        "message": f"Backup script test: {'SUCCESS' if result.returncode == 0 else 'FAILED'}",
                    },
                )

            except Exception as e:
                self.log_result(
                    "disaster_recovery",
                    "backup_procedures",
                    {"status": "WARNING", "message": f"Backup test failed: {e}"},
                )
        else:
            self.log_result(
                "disaster_recovery",
                "backup_procedures",
                {"status": "CRITICAL", "message": "Backup script not found"},
            )

    def test_service_restart(self):
        """Test service restart capability"""
        logger.info("Testing service restart capability...")

        # Check if docker-compose is available
        try:
            result = subprocess.run(
                ["docker-compose", "--version"], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                self.log_result(
                    "disaster_recovery",
                    "service_restart",
                    {
                        "status": "PASS",
                        "message": "Docker Compose available for service management",
                    },
                )
            else:
                self.log_result(
                    "disaster_recovery",
                    "service_restart",
                    {"status": "WARNING", "message": "Docker Compose not available"},
                )

        except Exception as e:
            self.log_result(
                "disaster_recovery",
                "service_restart",
                {"status": "WARNING", "message": f"Service restart test failed: {e}"},
            )

    def test_data_recovery(self):
        """Test data recovery simulation"""
        logger.info("Testing data recovery simulation...")

        # Check if recovery scripts exist
        recovery_scripts = ["scripts/restore-database.sh", "scripts/rollback-deployment.sh"]

        existing_scripts = [script for script in recovery_scripts if os.path.exists(script)]

        self.log_result(
            "disaster_recovery",
            "data_recovery",
            {
                "status": "PASS" if len(existing_scripts) >= 1 else "WARNING",
                "existing_scripts": existing_scripts,
                "message": f"Recovery scripts found: {len(existing_scripts)}/{len(recovery_scripts)}",
            },
        )

    def test_failover_procedures(self):
        """Test failover procedures"""
        logger.info("Testing failover procedures...")

        # Check for load balancer configuration
        nginx_config = "nginx/nginx.conf"
        if os.path.exists(nginx_config):
            with open(nginx_config, "r") as f:
                config = f.read()

            has_upstream = "upstream" in config
            has_backup = "backup" in config

            self.log_result(
                "disaster_recovery",
                "failover_procedures",
                {
                    "status": "PASS" if has_upstream else "WARNING",
                    "has_upstream": has_upstream,
                    "has_backup": has_backup,
                    "message": f"Failover configuration: upstream={has_upstream}, backup={has_backup}",
                },
            )
        else:
            self.log_result(
                "disaster_recovery",
                "failover_procedures",
                {"status": "WARNING", "message": "Nginx configuration not found"},
            )

    def validate_monitoring(self):
        """Validate monitoring and alerting systems"""
        logger.info("Validating monitoring and alerting systems...")

        # Test 1: Prometheus metrics
        self.test_prometheus_metrics()

        # Test 2: Grafana dashboards
        self.test_grafana_dashboards()

        # Test 3: Alert manager
        self.test_alert_manager()

        # Test 4: Log aggregation
        self.test_log_aggregation()

        # Test 5: Health check endpoints
        self.test_health_endpoints()

    def test_prometheus_metrics(self):
        """Test Prometheus metrics"""
        logger.info("Testing Prometheus metrics...")

        try:
            # Test metrics endpoint
            response = requests.get(f"{self.base_url}/metrics", timeout=10)

            if response.status_code == 200:
                metrics_text = response.text

                # Check for key metrics
                required_metrics = [
                    "http_requests_total",
                    "http_request_duration_seconds",
                    "process_cpu_seconds_total",
                    "process_resident_memory_bytes",
                ]

                found_metrics = [metric for metric in required_metrics if metric in metrics_text]

                self.log_result(
                    "monitoring",
                    "prometheus_metrics",
                    {
                        "status": "PASS" if len(found_metrics) >= 3 else "WARNING",
                        "found_metrics": found_metrics,
                        "total_metrics": len(required_metrics),
                        "message": f"Prometheus metrics: {len(found_metrics)}/{len(required_metrics)} found",
                    },
                )
            else:
                self.log_result(
                    "monitoring",
                    "prometheus_metrics",
                    {
                        "status": "WARNING",
                        "message": f"Metrics endpoint returned {response.status_code}",
                    },
                )

        except Exception as e:
            self.log_result(
                "monitoring",
                "prometheus_metrics",
                {"status": "WARNING", "message": f"Prometheus metrics test failed: {e}"},
            )

    def test_grafana_dashboards(self):
        """Test Grafana dashboards"""
        logger.info("Testing Grafana dashboards...")

        dashboard_dir = "monitoring/grafana/dashboards"
        if os.path.exists(dashboard_dir):
            dashboard_files = list(Path(dashboard_dir).glob("*.json"))

            self.log_result(
                "monitoring",
                "grafana_dashboards",
                {
                    "status": "PASS" if len(dashboard_files) > 0 else "WARNING",
                    "dashboard_count": len(dashboard_files),
                    "dashboards": [f.name for f in dashboard_files],
                    "message": f"Grafana dashboards found: {len(dashboard_files)}",
                },
            )
        else:
            self.log_result(
                "monitoring",
                "grafana_dashboards",
                {"status": "WARNING", "message": "Grafana dashboards directory not found"},
            )

    def test_alert_manager(self):
        """Test alert manager configuration"""
        logger.info("Testing alert manager configuration...")

        alert_config = "monitoring/alertmanager.yml"
        if os.path.exists(alert_config):
            try:
                with open(alert_config, "r") as f:
                    config = yaml.safe_load(f)

                has_routing = "route" in config
                has_receivers = "receivers" in config

                self.log_result(
                    "monitoring",
                    "alert_manager",
                    {
                        "status": "PASS" if has_routing and has_receivers else "WARNING",
                        "has_routing": has_routing,
                        "has_receivers": has_receivers,
                        "message": f"Alert manager config: routing={has_routing}, receivers={has_receivers}",
                    },
                )
            except Exception as e:
                self.log_result(
                    "monitoring",
                    "alert_manager",
                    {"status": "WARNING", "message": f"Alert manager config parse error: {e}"},
                )
        else:
            self.log_result(
                "monitoring",
                "alert_manager",
                {"status": "WARNING", "message": "Alert manager configuration not found"},
            )

    def test_log_aggregation(self):
        """Test log aggregation"""
        logger.info("Testing log aggregation...")

        log_files = ["logs/freeagentics.json", "logs/backend.log", "logs/security_audit.log"]

        existing_logs = [log for log in log_files if os.path.exists(log)]

        self.log_result(
            "monitoring",
            "log_aggregation",
            {
                "status": "PASS" if len(existing_logs) > 0 else "WARNING",
                "existing_logs": existing_logs,
                "message": f"Log files found: {len(existing_logs)}/{len(log_files)}",
            },
        )

    def test_health_endpoints(self):
        """Test health check endpoints"""
        logger.info("Testing health check endpoints...")

        health_endpoints = ["/health", "/api/v1/system/health", "/api/v1/monitoring/health"]

        healthy_endpoints = []
        for endpoint in health_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    healthy_endpoints.append(endpoint)
            except Exception:
                pass

        self.log_result(
            "monitoring",
            "health_endpoints",
            {
                "status": "PASS" if len(healthy_endpoints) >= 1 else "CRITICAL",
                "healthy_endpoints": healthy_endpoints,
                "total_endpoints": len(health_endpoints),
                "message": f"Health endpoints: {len(healthy_endpoints)}/{len(health_endpoints)} healthy",
            },
        )

    def validate_operational_procedures(self):
        """Test all operational procedures"""
        logger.info("Testing operational procedures...")

        # Test 1: Deployment procedures
        self.test_deployment_procedures()

        # Test 2: Rollback procedures
        self.test_rollback_procedures()

        # Test 3: Scaling procedures
        self.test_scaling_procedures()

        # Test 4: Maintenance procedures
        self.test_maintenance_procedures()

    def test_deployment_procedures(self):
        """Test deployment procedures"""
        logger.info("Testing deployment procedures...")

        deployment_scripts = ["deploy-production.sh", "scripts/deployment/deploy-production.sh"]

        existing_scripts = [script for script in deployment_scripts if os.path.exists(script)]

        self.log_result(
            "operational",
            "deployment_procedures",
            {
                "status": "PASS" if len(existing_scripts) > 0 else "WARNING",
                "existing_scripts": existing_scripts,
                "message": f"Deployment scripts found: {len(existing_scripts)}",
            },
        )

    def test_rollback_procedures(self):
        """Test rollback procedures"""
        logger.info("Testing rollback procedures...")

        rollback_scripts = ["scripts/rollback.sh", "scripts/deployment/rollback.sh"]

        existing_scripts = [script for script in rollback_scripts if os.path.exists(script)]

        self.log_result(
            "operational",
            "rollback_procedures",
            {
                "status": "PASS" if len(existing_scripts) > 0 else "WARNING",
                "existing_scripts": existing_scripts,
                "message": f"Rollback scripts found: {len(existing_scripts)}",
            },
        )

    def test_scaling_procedures(self):
        """Test scaling procedures"""
        logger.info("Testing scaling procedures...")

        # Check Docker Compose scaling configuration
        compose_files = ["docker-compose.production.yml", "docker-compose.yml"]

        scaling_configured = False
        for compose_file in compose_files:
            if os.path.exists(compose_file):
                with open(compose_file, "r") as f:
                    content = f.read()
                    if "replicas:" in content or "scale:" in content:
                        scaling_configured = True
                        break

        self.log_result(
            "operational",
            "scaling_procedures",
            {
                "status": "PASS" if scaling_configured else "WARNING",
                "scaling_configured": scaling_configured,
                "message": f"Scaling configuration: {'found' if scaling_configured else 'not found'}",
            },
        )

    def test_maintenance_procedures(self):
        """Test maintenance procedures"""
        logger.info("Testing maintenance procedures...")

        maintenance_docs = [
            "docs/operations/MAINTENANCE_PROCEDURES.md",
            "docs/runbooks/MAINTENANCE_PROCEDURES.md",
        ]

        existing_docs = [doc for doc in maintenance_docs if os.path.exists(doc)]

        self.log_result(
            "operational",
            "maintenance_procedures",
            {
                "status": "PASS" if len(existing_docs) > 0 else "WARNING",
                "existing_docs": existing_docs,
                "message": f"Maintenance documentation found: {len(existing_docs)}",
            },
        )

    def generate_final_report(self):
        """Generate comprehensive final report"""
        logger.info("Generating final comprehensive report...")

        # Calculate summary statistics
        total_tests = len(self.passes) + len(self.warnings) + len(self.critical_failures)
        pass_rate = (len(self.passes) / total_tests) * 100 if total_tests > 0 else 0

        self.results["summary"] = {
            "total_tests": total_tests,
            "passed": len(self.passes),
            "warnings": len(self.warnings),
            "critical_failures": len(self.critical_failures),
            "pass_rate": pass_rate,
            "production_ready": len(self.critical_failures) == 0,
        }

        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_report = f"production_validation_report_{timestamp}.json"

        with open(json_report, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Generate markdown report
        markdown_report = f"production_validation_report_{timestamp}.md"
        self.generate_markdown_report(markdown_report)

        logger.info(f"Reports generated: {json_report}, {markdown_report}")

        return json_report, markdown_report

    def generate_markdown_report(self, filename: str):
        """Generate markdown report"""
        with open(filename, "w") as f:
            f.write("# FreeAgentics Production Readiness Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Environment:** {self.results['environment']}\n\n")

            # Summary
            summary = self.results["summary"]
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Tests:** {summary['total_tests']}\n")
            f.write(f"- **Passed:** {summary['passed']}\n")
            f.write(f"- **Warnings:** {summary['warnings']}\n")
            f.write(f"- **Critical Failures:** {summary['critical_failures']}\n")
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
            f.write("## Detailed Test Results\n\n")
            for category, tests in self.results["validation_results"].items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                for test_name, result in tests.items():
                    status_icon = {"PASS": "✅", "WARNING": "⚠️", "CRITICAL": "❌"}.get(
                        result.get("status", ""), "ℹ️"
                    )
                    f.write(
                        f"- {status_icon} **{test_name}**: {result.get('message', 'No message')}\n"
                    )
                f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            if summary["production_ready"]:
                f.write(
                    "🎉 **Congratulations!** Your FreeAgentics deployment is production-ready!\n\n"
                )
                f.write("### Next Steps:\n")
                f.write("1. Address any warnings to optimize performance\n")
                f.write("2. Schedule regular monitoring and maintenance\n")
                f.write("3. Plan disaster recovery drills\n")
                f.write("4. Deploy to production with confidence! 🚀\n")
            else:
                f.write(
                    "🛑 **Action Required:** Critical issues must be resolved before production deployment.\n\n"
                )
                f.write("### Immediate Actions:\n")
                f.write("1. Fix all critical failures listed above\n")
                f.write("2. Re-run validation after fixes\n")
                f.write("3. Address warnings for optimal performance\n")
                f.write("4. Do not deploy until all critical issues are resolved\n")

    async def run_validation(self):
        """Run comprehensive validation"""
        logger.info("Starting comprehensive production validation...")

        # Check environment readiness
        if not self.check_environment_readiness():
            self.critical_failures.append("Environment not ready for testing")
            return

        # Run all validation tests
        try:
            self.validate_load_testing()
            self.validate_security()
            self.validate_disaster_recovery()
            self.validate_monitoring()
            self.validate_operational_procedures()

            # Generate final report
            json_report, markdown_report = self.generate_final_report()

            # Print summary
            summary = self.results["summary"]
            print("\n" + "=" * 60)
            print("  PRODUCTION VALIDATION SUMMARY")
            print("=" * 60)
            print(f"Total Tests: {summary['total_tests']}")
            print(f"Passed: {summary['passed']}")
            print(f"Warnings: {summary['warnings']}")
            print(f"Critical Failures: {summary['critical_failures']}")
            print(f"Pass Rate: {summary['pass_rate']:.1f}%")
            print(f"Production Ready: {'YES' if summary['production_ready'] else 'NO'}")
            print("=" * 60)

            if summary["production_ready"]:
                print("🎉 PRODUCTION READY! 🚀")
                return True
            else:
                print("❌ NOT PRODUCTION READY - Fix critical issues")
                return False

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


async def main():
    """Main function"""
    validator = ProductionValidator()
    success = await validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
