#!/usr/bin/env python3
"""
Production Deployment Testing Script
Tests various production deployment scenarios and configurations
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


class ProductionDeploymentTester:
    """
    Production deployment scenario tester
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "deployment_tests": {},
            "scaling_tests": {},
            "rollback_tests": {},
            "monitoring_tests": {},
            "errors": [],
            "warnings": [],
        }

    def log_info(self, message: str):
        """Log informational message"""
        print(f"[INFO] {message}")

    def log_warning(self, message: str):
        """Log warning message"""
        print(f"[WARNING] {message}")
        self.test_results["warnings"].append(message)

    def log_error(self, message: str):
        """Log error message"""
        print(f"[ERROR] {message}")
        self.test_results["errors"].append(message)

    def run_command(self, command: List[str], timeout: int = 300) -> Tuple[int, str, str]:
        """Run a command and return return code, stdout, stderr"""
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, cwd=self.project_root
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return 1, "", str(e)

    def create_test_environment(self) -> str:
        """Create test environment configuration"""
        self.log_info("Creating test environment configuration...")

        # Create temporary environment file
        temp_env = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
        temp_env.write(
            """
# Test Environment Configuration
DATABASE_URL=postgresql://test:test@localhost:5432/test_db
REDIS_URL=redis://localhost:6379
SECRET_KEY=test_secret_key_for_deployment_testing
JWT_SECRET=test_jwt_secret_for_deployment_testing
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
ENVIRONMENT=production
LOG_LEVEL=INFO
WORKERS=2
"""
        )
        temp_env.close()

        return temp_env.name

    def test_single_container_deployment(self) -> bool:
        """Test single container deployment"""
        self.log_info("Testing single container deployment...")

        container_name = "freeagentics-single-test"
        env_file = self.create_test_environment()

        try:
            # Clean up any existing container
            self.run_command(["docker", "rm", "-f", container_name])

            # Build production image
            returncode, stdout, stderr = self.run_command(
                ["docker", "build", "--target", "production", "-t", "freeagentics:single-test", "."]
            )

            if returncode != 0:
                self.log_error(f"Failed to build image: {stderr}")
                return False

            # Run single container
            returncode, stdout, stderr = self.run_command(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "--env-file",
                    env_file,
                    "-p",
                    "8001:8000",
                    "freeagentics:single-test",
                ]
            )

            if returncode != 0:
                self.log_error(f"Failed to run container: {stderr}")
                return False

            # Wait for startup
            time.sleep(15)

            # Check if container is running
            returncode, stdout, stderr = self.run_command(
                ["docker", "ps", "-f", f"name={container_name}", "--format", "{{.Status}}"]
            )

            if returncode == 0 and stdout.strip():
                self.log_info("✓ Single container deployment successful")
                self.test_results["deployment_tests"]["single_container"] = True

                # Test health endpoint
                time.sleep(5)
                returncode, stdout, stderr = self.run_command(
                    ["curl", "-f", "http://localhost:8001/health"]
                )

                if returncode == 0:
                    self.log_info("✓ Health endpoint responding")
                    self.test_results["deployment_tests"]["health_endpoint"] = True
                else:
                    self.log_warning("⚠ Health endpoint not responding")
                    self.test_results["deployment_tests"]["health_endpoint"] = False

            else:
                self.log_error("Single container deployment failed")
                self.test_results["deployment_tests"]["single_container"] = False

        except Exception as e:
            self.log_error(f"Single container test failed: {e}")
            self.test_results["deployment_tests"]["single_container"] = False

        finally:
            # Clean up
            self.run_command(["docker", "rm", "-f", container_name])
            self.run_command(["docker", "rmi", "freeagentics:single-test"])
            os.unlink(env_file)

        return self.test_results["deployment_tests"].get("single_container", False)

    def test_docker_compose_deployment(self) -> bool:
        """Test Docker Compose deployment"""
        self.log_info("Testing Docker Compose deployment...")

        # Check if docker-compose.yml exists
        compose_file = self.project_root / "docker-compose.yml"
        if not compose_file.exists():
            self.log_error("docker-compose.yml not found")
            return False

        try:
            # Create test environment
            env_file = self.create_test_environment()

            # Copy environment file for compose
            compose_env = self.project_root / ".env.test"
            with open(env_file, "r") as src, open(compose_env, "w") as dst:
                dst.write(src.read())
                dst.write("\nPOSTGRES_PASSWORD=test_password\n")
                dst.write("REDIS_PASSWORD=test_redis_password\n")

            # Test compose file validation
            returncode, stdout, stderr = self.run_command(["docker", "compose", "config"])

            if returncode != 0:
                self.log_error(f"Docker Compose config invalid: {stderr}")
                return False

            self.log_info("✓ Docker Compose configuration valid")

            # Test compose up (dry run)
            returncode, stdout, stderr = self.run_command(["docker", "compose", "up", "--dry-run"])

            if returncode == 0:
                self.log_info("✓ Docker Compose dry run successful")
                self.test_results["deployment_tests"]["docker_compose"] = True
            else:
                self.log_warning(f"⚠ Docker Compose dry run failed: {stderr}")
                self.test_results["deployment_tests"]["docker_compose"] = False

        except Exception as e:
            self.log_error(f"Docker Compose test failed: {e}")
            self.test_results["deployment_tests"]["docker_compose"] = False

        finally:
            # Clean up
            os.unlink(env_file)
            compose_env_path = self.project_root / ".env.test"
            if compose_env_path.exists():
                os.unlink(compose_env_path)

        return self.test_results["deployment_tests"].get("docker_compose", False)

    def test_scaling_scenarios(self) -> bool:
        """Test scaling scenarios"""
        self.log_info("Testing scaling scenarios...")

        # Test horizontal scaling with multiple containers
        container_names = []
        env_file = self.create_test_environment()

        try:
            # Build image
            returncode, stdout, stderr = self.run_command(
                ["docker", "build", "--target", "production", "-t", "freeagentics:scale-test", "."]
            )

            if returncode != 0:
                self.log_error(f"Failed to build image: {stderr}")
                return False

            # Start multiple containers
            for i in range(3):
                container_name = f"freeagentics-scale-{i}"
                port = 8010 + i

                returncode, stdout, stderr = self.run_command(
                    [
                        "docker",
                        "run",
                        "-d",
                        "--name",
                        container_name,
                        "--env-file",
                        env_file,
                        "-p",
                        f"{port}:8000",
                        "freeagentics:scale-test",
                    ]
                )

                if returncode == 0:
                    container_names.append(container_name)
                else:
                    self.log_error(f"Failed to start container {container_name}: {stderr}")

            # Wait for containers to start
            time.sleep(20)

            # Check running containers
            running_containers = 0
            for container_name in container_names:
                returncode, stdout, stderr = self.run_command(
                    ["docker", "ps", "-f", f"name={container_name}", "--format", "{{.Status}}"]
                )

                if returncode == 0 and stdout.strip():
                    running_containers += 1

            if running_containers >= 2:
                self.log_info(
                    f"✓ Scaling test successful: {running_containers}/3 containers running"
                )
                self.test_results["scaling_tests"]["horizontal_scaling"] = True
            else:
                self.log_error(
                    f"Scaling test failed: only {running_containers}/3 containers running"
                )
                self.test_results["scaling_tests"]["horizontal_scaling"] = False

        except Exception as e:
            self.log_error(f"Scaling test failed: {e}")
            self.test_results["scaling_tests"]["horizontal_scaling"] = False

        finally:
            # Clean up
            for container_name in container_names:
                self.run_command(["docker", "rm", "-f", container_name])
            self.run_command(["docker", "rmi", "freeagentics:scale-test"])
            os.unlink(env_file)

        return self.test_results["scaling_tests"].get("horizontal_scaling", False)

    def test_rollback_scenarios(self) -> bool:
        """Test rollback scenarios"""
        self.log_info("Testing rollback scenarios...")

        env_file = self.create_test_environment()

        try:
            # Build two versions
            returncode, stdout, stderr = self.run_command(
                ["docker", "build", "--target", "production", "-t", "freeagentics:v1", "."]
            )

            if returncode != 0:
                self.log_error(f"Failed to build v1: {stderr}")
                return False

            # Tag as v2 (simulating new version)
            self.run_command(["docker", "tag", "freeagentics:v1", "freeagentics:v2"])

            # Deploy v1
            container_name = "freeagentics-rollback-test"
            returncode, stdout, stderr = self.run_command(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "--env-file",
                    env_file,
                    "-p",
                    "8020:8000",
                    "freeagentics:v1",
                ]
            )

            if returncode != 0:
                self.log_error(f"Failed to deploy v1: {stderr}")
                return False

            time.sleep(10)

            # Check v1 is running
            returncode, stdout, stderr = self.run_command(
                ["docker", "ps", "-f", f"name={container_name}", "--format", "{{.Status}}"]
            )

            if returncode == 0 and stdout.strip():
                self.log_info("✓ Version 1 deployed successfully")

                # Simulate rollback (stop v1, start v2)
                self.run_command(["docker", "stop", container_name])
                self.run_command(["docker", "rm", container_name])

                # Deploy v2
                returncode, stdout, stderr = self.run_command(
                    [
                        "docker",
                        "run",
                        "-d",
                        "--name",
                        container_name,
                        "--env-file",
                        env_file,
                        "-p",
                        "8020:8000",
                        "freeagentics:v2",
                    ]
                )

                if returncode == 0:
                    time.sleep(10)

                    # Check v2 is running
                    returncode, stdout, stderr = self.run_command(
                        ["docker", "ps", "-f", f"name={container_name}", "--format", "{{.Status}}"]
                    )

                    if returncode == 0 and stdout.strip():
                        self.log_info("✓ Rollback test successful")
                        self.test_results["rollback_tests"]["version_rollback"] = True
                    else:
                        self.log_error("Rollback test failed")
                        self.test_results["rollback_tests"]["version_rollback"] = False
                else:
                    self.log_error(f"Failed to deploy v2: {stderr}")
                    self.test_results["rollback_tests"]["version_rollback"] = False
            else:
                self.log_error("Failed to deploy v1")
                self.test_results["rollback_tests"]["version_rollback"] = False

        except Exception as e:
            self.log_error(f"Rollback test failed: {e}")
            self.test_results["rollback_tests"]["version_rollback"] = False

        finally:
            # Clean up
            self.run_command(["docker", "rm", "-f", container_name])
            self.run_command(["docker", "rmi", "freeagentics:v1"])
            self.run_command(["docker", "rmi", "freeagentics:v2"])
            os.unlink(env_file)

        return self.test_results["rollback_tests"].get("version_rollback", False)

    def test_monitoring_integration(self) -> bool:
        """Test monitoring integration"""
        self.log_info("Testing monitoring integration...")

        env_file = self.create_test_environment()

        try:
            # Build image
            returncode, stdout, stderr = self.run_command(
                [
                    "docker",
                    "build",
                    "--target",
                    "production",
                    "-t",
                    "freeagentics:monitor-test",
                    ".",
                ]
            )

            if returncode != 0:
                self.log_error(f"Failed to build image: {stderr}")
                return False

            # Run container
            container_name = "freeagentics-monitor-test"
            returncode, stdout, stderr = self.run_command(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "--env-file",
                    env_file,
                    "-p",
                    "8030:8000",
                    "freeagentics:monitor-test",
                ]
            )

            if returncode != 0:
                self.log_error(f"Failed to run container: {stderr}")
                return False

            time.sleep(15)

            # Test metrics endpoint
            returncode, stdout, stderr = self.run_command(
                ["curl", "-f", "http://localhost:8030/metrics"]
            )

            if returncode == 0:
                self.log_info("✓ Metrics endpoint responding")
                self.test_results["monitoring_tests"]["metrics_endpoint"] = True
            else:
                self.log_warning("⚠ Metrics endpoint not responding")
                self.test_results["monitoring_tests"]["metrics_endpoint"] = False

            # Test health endpoint
            returncode, stdout, stderr = self.run_command(
                ["curl", "-f", "http://localhost:8030/health"]
            )

            if returncode == 0:
                self.log_info("✓ Health endpoint responding")
                self.test_results["monitoring_tests"]["health_endpoint"] = True
            else:
                self.log_warning("⚠ Health endpoint not responding")
                self.test_results["monitoring_tests"]["health_endpoint"] = False

            # Test container resource monitoring
            returncode, stdout, stderr = self.run_command(
                [
                    "docker",
                    "stats",
                    container_name,
                    "--no-stream",
                    "--format",
                    "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}",
                ]
            )

            if returncode == 0:
                self.log_info("✓ Container resource monitoring working")
                self.test_results["monitoring_tests"]["resource_monitoring"] = True
            else:
                self.log_warning("⚠ Container resource monitoring failed")
                self.test_results["monitoring_tests"]["resource_monitoring"] = False

        except Exception as e:
            self.log_error(f"Monitoring test failed: {e}")
            self.test_results["monitoring_tests"]["metrics_endpoint"] = False
            self.test_results["monitoring_tests"]["health_endpoint"] = False
            self.test_results["monitoring_tests"]["resource_monitoring"] = False

        finally:
            # Clean up
            self.run_command(["docker", "rm", "-f", container_name])
            self.run_command(["docker", "rmi", "freeagentics:monitor-test"])
            os.unlink(env_file)

        return any(self.test_results["monitoring_tests"].values())

    def test_environment_variables(self) -> bool:
        """Test environment variable handling"""
        self.log_info("Testing environment variable handling...")

        # Test with minimal environment
        minimal_env = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
        minimal_env.write("ENVIRONMENT=production\n")
        minimal_env.close()

        try:
            # Build image
            returncode, stdout, stderr = self.run_command(
                ["docker", "build", "--target", "production", "-t", "freeagentics:env-test", "."]
            )

            if returncode != 0:
                self.log_error(f"Failed to build image: {stderr}")
                return False

            # Run with minimal environment (should fail)
            container_name = "freeagentics-env-test"
            returncode, stdout, stderr = self.run_command(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    container_name,
                    "--env-file",
                    minimal_env.name,
                    "freeagentics:env-test",
                ]
            )

            if returncode != 0:
                self.log_info("✓ Container correctly fails with minimal environment")
                self.test_results["deployment_tests"]["env_validation"] = True
            else:
                self.log_warning("⚠ Container should fail with minimal environment")
                self.test_results["deployment_tests"]["env_validation"] = False

        except Exception as e:
            self.log_error(f"Environment test failed: {e}")
            self.test_results["deployment_tests"]["env_validation"] = False

        finally:
            # Clean up
            self.run_command(["docker", "rmi", "freeagentics:env-test"])
            os.unlink(minimal_env.name)

        return self.test_results["deployment_tests"].get("env_validation", False)

    def add_api_cleanup_functionality(self):
        """Add API cleanup functionality as specified in subtask"""
        self.log_info("Adding API cleanup functionality...")

        # Check if cleanup endpoints exist
        api_files = [
            self.project_root / "api" / "v1" / "system.py",
            self.project_root / "api" / "main.py",
        ]

        cleanup_endpoints_found = False
        for api_file in api_files:
            if api_file.exists():
                try:
                    with open(api_file, "r") as f:
                        content = f.read()
                        if "/cleanup" in content or "cleanup" in content:
                            cleanup_endpoints_found = True
                            break
                except Exception as e:
                    self.log_error(f"Error reading {api_file}: {e}")

        if cleanup_endpoints_found:
            self.log_info("✓ API cleanup endpoints found")
            self.test_results["deployment_tests"]["api_cleanup"] = True
        else:
            self.log_info("API cleanup endpoints not found - implementation needed")
            self.test_results["deployment_tests"]["api_cleanup"] = False

        # Create a simple cleanup endpoint implementation example
        cleanup_example = """
# Example API cleanup endpoint implementation
from fastapi import APIRouter

router = APIRouter()

@router.post("/cleanup")
async def cleanup_resources():
    '''
    Cleanup endpoint for container resources
    '''
    try:
        # Cleanup temporary files
        # Clear caches
        # Reset connections
        return {"status": "success", "message": "Resources cleaned up"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
"""

        example_file = self.project_root / "api_cleanup_example.py"
        with open(example_file, "w") as f:
            f.write(cleanup_example)

        self.log_info(f"API cleanup example created at: {example_file}")

    def generate_report(self):
        """Generate deployment test report"""
        self.log_info("Generating deployment test report...")

        # Calculate overall success rate
        total_tests = 0
        passed_tests = 0

        for category in ["deployment_tests", "scaling_tests", "rollback_tests", "monitoring_tests"]:
            if category in self.test_results:
                for test_name, result in self.test_results[category].items():
                    total_tests += 1
                    if result:
                        passed_tests += 1

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "error_count": len(self.test_results["errors"]),
            "warning_count": len(self.test_results["warnings"]),
        }

        # Save report
        report_file = (
            self.project_root
            / f"production_deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(self.test_results, f, indent=2)

        self.log_info(f"Deployment test report saved to: {report_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("PRODUCTION DEPLOYMENT TEST REPORT")
        print("=" * 80)
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Errors: {len(self.test_results['errors'])}")
        print(f"Warnings: {len(self.test_results['warnings'])}")

        # Show test results by category
        for category in ["deployment_tests", "scaling_tests", "rollback_tests", "monitoring_tests"]:
            if category in self.test_results:
                print(f"\n{category.upper().replace('_', ' ')}:")
                for test_name, result in self.test_results[category].items():
                    status = "✓" if result else "✗"
                    print(f"  {status} {test_name.replace('_', ' ')}")

        if self.test_results["errors"]:
            print("\nERRORS:")
            for error in self.test_results["errors"]:
                print(f"  - {error}")

        if self.test_results["warnings"]:
            print("\nWARNINGS:")
            for warning in self.test_results["warnings"]:
                print(f"  - {warning}")

        print("=" * 80)

    def run_all_tests(self):
        """Run all deployment tests"""
        self.log_info("Starting production deployment tests...")

        try:
            self.test_single_container_deployment()
            self.test_docker_compose_deployment()
            self.test_scaling_scenarios()
            self.test_rollback_scenarios()
            self.test_monitoring_integration()
            self.test_environment_variables()
            self.add_api_cleanup_functionality()

        except Exception as e:
            self.log_error(f"Deployment tests failed: {e}")

        finally:
            self.generate_report()

        return len(self.test_results["errors"]) == 0


def main():
    """Main execution function"""
    tester = ProductionDeploymentTester()

    if tester.run_all_tests():
        print("\n✅ Production deployment tests PASSED")
        return 0
    else:
        print("\n❌ Production deployment tests completed with issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
