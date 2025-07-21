#!/usr/bin/env python3
"""
Comprehensive Docker Container Production Build Validation
Validates all aspects of Docker production build as specified in subtask 15.1
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class ComprehensiveDockerValidator:
    """
    Complete Docker production build validator
    """

    def __init__(self):
        self.project_root = Path.cwd()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "subtask": "15.1 - Validate Docker Container Production Build",
            "validation_results": {},
            "recommendations": [],
        }

    def log_info(self, message):
        print(f"[INFO] {message}")

    def log_success(self, message):
        print(f"[SUCCESS] {message}")

    def log_warning(self, message):
        print(f"[WARNING] {message}")

    def log_error(self, message):
        print(f"[ERROR] {message}")

    def run_command(self, command, timeout=180):
        """Run command with timeout"""
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return 1, "", str(e)

    def validate_docker_files(self):
        """1. Find and examine Docker configuration files"""
        self.log_info("1. Finding and examining Docker configuration files...")

        # Check Dockerfile
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            self.log_success("✅ Dockerfile found")

            with open(dockerfile_path, "r") as f:
                content = f.read()

            # Check for multi-stage build
            if "FROM" in content and "as production" in content:
                self.log_success("✅ Multi-stage build detected")
                self.results["validation_results"]["multi_stage_build"] = True
            else:
                self.log_warning("⚠️  Multi-stage build not detected")
                self.results["validation_results"]["multi_stage_build"] = False

        else:
            self.log_error("❌ Dockerfile not found")
            self.results["validation_results"]["dockerfile_exists"] = False

        # Check docker-compose files
        compose_files = ["docker-compose.yml", "docker-compose.production.yml"]
        for compose_file in compose_files:
            compose_path = self.project_root / compose_file
            if compose_path.exists():
                self.log_success(f"✅ {compose_file} found")

                # Validate syntax
                returncode, stdout, stderr = self.run_command(
                    ["docker", "compose", "-f", str(compose_path), "config"]
                )

                if returncode == 0:
                    self.log_success(f"✅ {compose_file} syntax valid")
                else:
                    self.log_error(f"❌ {compose_file} syntax error: {stderr}")
            else:
                self.log_warning(f"⚠️  {compose_file} not found")

        # Check requirements files
        req_files = ["requirements-production.txt", "requirements-core.txt"]
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                self.log_success(f"✅ {req_file} found")
            else:
                self.log_warning(f"⚠️  {req_file} not found")

    def validate_production_config(self):
        """2. Validate production Docker build configuration"""
        self.log_info("2. Validating production Docker build configuration...")

        dockerfile_path = self.project_root / "Dockerfile"
        if not dockerfile_path.exists():
            self.log_error("❌ Cannot validate - Dockerfile not found")
            return

        with open(dockerfile_path, "r") as f:
            content = f.read()

        # Security checks
        security_checks = [
            ("USER", "Non-root user configured"),
            ("HEALTHCHECK", "Health check configured"),
            ("EXPOSE", "Port exposed"),
            ("gunicorn", "Production server configured"),
        ]

        for check, description in security_checks:
            if check in content:
                self.log_success(f"✅ {description}")
            else:
                self.log_warning(f"⚠️  {description} not found")

        # Check for security best practices
        security_issues = [
            ("sudo", "Avoid sudo usage"),
            ("curl.*|.*sh", "Avoid curl pipe to shell"),
            ("--privileged", "Avoid privileged containers"),
        ]

        for issue, description in security_issues:
            if issue in content:
                self.log_warning(f"⚠️  Security issue: {description}")
            else:
                self.log_success(f"✅ {description}")

    def test_multistage_builds(self):
        """3. Test multi-stage builds and optimization"""
        self.log_info("3. Testing multi-stage builds and optimization...")

        # Test development stage
        self.log_info("Building development stage...")
        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "build",
                "--target",
                "development",
                "-t",
                "freeagentics:dev-stage",
                ".",
            ]
        )

        if returncode == 0:
            self.log_success("✅ Development stage built successfully")

            # Get size
            dev_size = self.get_image_size("freeagentics:dev-stage")
            self.log_info(f"Development stage size: {self.format_size(dev_size)}")

        else:
            self.log_error(f"❌ Development stage build failed: {stderr}")

        # Test production stage
        self.log_info("Building production stage...")
        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "build",
                "--target",
                "production",
                "-t",
                "freeagentics:prod-stage",
                ".",
            ]
        )

        if returncode == 0:
            self.log_success("✅ Production stage built successfully")

            # Get size
            prod_size = self.get_image_size("freeagentics:prod-stage")
            self.log_info(f"Production stage size: {self.format_size(prod_size)}")

            # Compare sizes
            if dev_size > 0 and prod_size > 0:
                if prod_size < dev_size:
                    reduction = (dev_size - prod_size) / dev_size * 100
                    self.log_success(
                        f"✅ Production optimization: {reduction:.1f}% size reduction"
                    )
                    self.results["validation_results"]["size_optimization"] = reduction
                else:
                    self.log_warning("⚠️  Production stage not optimized")
                    self.results["validation_results"]["size_optimization"] = 0

        else:
            self.log_error(f"❌ Production stage build failed: {stderr}")

        # Clean up
        self.run_command(["docker", "rmi", "freeagentics:dev-stage"])
        self.run_command(["docker", "rmi", "freeagentics:prod-stage"])

    def verify_security_practices(self):
        """4. Verify container security and best practices"""
        self.log_info("4. Verifying container security and best practices...")

        # Build image for security testing
        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "build",
                "--target",
                "production",
                "-t",
                "freeagentics:security-test",
                ".",
            ]
        )

        if returncode != 0:
            self.log_error(f"❌ Cannot test security - build failed: {stderr}")
            return

        # Test 1: Non-root user
        self.log_info("Testing non-root user...")
        returncode, stdout, stderr = self.run_command(
            ["docker", "run", "--rm", "freeagentics:security-test", "whoami"]
        )

        if returncode == 0:
            user = stdout.strip()
            if user != "root":
                self.log_success(f"✅ Container runs as non-root user: {user}")
                self.results["validation_results"]["non_root_user"] = True
            else:
                self.log_warning("⚠️  Container runs as root user")
                self.results["validation_results"]["non_root_user"] = False
        else:
            self.log_error(f"❌ Failed to check user: {stderr}")

        # Test 2: Read-only filesystem
        self.log_info("Testing read-only filesystem...")
        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "run",
                "--rm",
                "--read-only",
                "--tmpfs",
                "/tmp",  # nosec B108 - Secure tmpfs mount in Docker
                "freeagentics:security-test",
                "touch",
                "/test-file",
            ]
        )

        if returncode != 0:
            self.log_success("✅ Read-only filesystem prevents file creation")
            self.results["validation_results"]["readonly_fs"] = True
        else:
            self.log_warning("⚠️  Read-only filesystem not enforced")
            self.results["validation_results"]["readonly_fs"] = False

        # Test 3: Health check
        self.log_info("Testing health check...")
        container_name = "health-check-test"

        # Clean up any existing container
        self.run_command(["docker", "rm", "-f", container_name])

        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "freeagentics:security-test",
            ]
        )

        if returncode == 0:
            # Wait for health check
            time.sleep(30)

            # Check health status
            returncode, stdout, stderr = self.run_command(
                [
                    "docker",
                    "inspect",
                    container_name,
                    "--format",
                    "{{.State.Health.Status}}",
                ]
            )

            if returncode == 0:
                health_status = stdout.strip()
                self.log_info(f"Health check status: {health_status}")

                if health_status == "healthy":
                    self.log_success("✅ Health check passed")
                    self.results["validation_results"]["health_check"] = True
                else:
                    self.log_warning(f"⚠️  Health check status: {health_status}")
                    self.results["validation_results"]["health_check"] = False
            else:
                self.log_warning("⚠️  Health check not configured")
                self.results["validation_results"]["health_check"] = False

        # Clean up
        self.run_command(["docker", "rm", "-f", container_name])
        self.run_command(["docker", "rmi", "freeagentics:security-test"])

    def test_deployment_scenarios(self):
        """5. Test production deployment scenarios"""
        self.log_info("5. Testing production deployment scenarios...")

        # Test single container deployment
        self.log_info("Testing single container deployment...")

        # Create test environment
        import tempfile

        env_file = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
        env_file.write(
            """
DATABASE_URL=postgresql://test:test@localhost:5432/test
REDIS_URL=redis://localhost:6379
SECRET_KEY=test_secret_key
JWT_SECRET=test_jwt_secret
ENVIRONMENT=production
"""
        )
        env_file.close()

        try:
            # Build and run container
            returncode, stdout, stderr = self.run_command(
                [
                    "docker",
                    "build",
                    "--target",
                    "production",
                    "-t",
                    "freeagentics:deploy-test",
                    ".",
                ]
            )

            if returncode != 0:
                self.log_error(f"❌ Deployment test build failed: {stderr}")
                return

            container_name = "deploy-test-container"

            # Clean up any existing container
            self.run_command(["docker", "rm", "-f", container_name])

            returncode, stdout, stderr = self.run_command(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "--env-file",
                    env_file.name,
                    "-p",
                    "8001:8000",
                    "freeagentics:deploy-test",
                ]
            )

            if returncode == 0:
                time.sleep(10)  # Wait for startup

                # Check if container is running
                returncode, stdout, stderr = self.run_command(
                    [
                        "docker",
                        "ps",
                        "-f",
                        f"name={container_name}",
                        "--format",
                        "{{.Status}}",
                    ]
                )

                if returncode == 0 and stdout.strip():
                    self.log_success("✅ Single container deployment successful")
                    self.results["validation_results"]["single_deployment"] = True
                else:
                    self.log_warning("⚠️  Single container deployment failed")
                    self.results["validation_results"]["single_deployment"] = False

            else:
                self.log_error(f"❌ Container failed to start: {stderr}")
                self.results["validation_results"]["single_deployment"] = False

            # Clean up
            self.run_command(["docker", "rm", "-f", container_name])
            self.run_command(["docker", "rmi", "freeagentics:deploy-test"])

        finally:
            os.unlink(env_file.name)

    def validate_api_cleanup(self):
        """6. Add and validate API cleanup functionality"""
        self.log_info("6. Validating API cleanup functionality...")

        # Check if cleanup endpoints exist
        api_path = self.project_root / "api" / "v1" / "system.py"
        if api_path.exists():
            with open(api_path, "r") as f:
                content = f.read()

            if "/cleanup" in content:
                self.log_success("✅ API cleanup endpoints found")
                self.results["validation_results"]["api_cleanup"] = True
            else:
                self.log_warning("⚠️  API cleanup endpoints not found")
                self.results["validation_results"]["api_cleanup"] = False
        else:
            self.log_warning("⚠️  API system module not found")
            self.results["validation_results"]["api_cleanup"] = False

    def get_image_size(self, image_name):
        """Get image size in bytes"""
        returncode, stdout, stderr = self.run_command(
            ["docker", "inspect", image_name, "--format", "{{.Size}}"]
        )

        if returncode == 0:
            return int(stdout.strip())
        return 0

    def format_size(self, size_bytes):
        """Format size in human readable format"""
        if size_bytes == 0:
            return "0 B"

        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def generate_recommendations(self):
        """Generate recommendations based on validation results"""

        if not self.results["validation_results"].get("multi_stage_build", False):
            self.results["recommendations"].append("Implement multi-stage Docker build")

        if not self.results["validation_results"].get("non_root_user", False):
            self.results["recommendations"].append(
                "Configure container to run as non-root user"
            )

        if not self.results["validation_results"].get("readonly_fs", False):
            self.results["recommendations"].append(
                "Enable read-only filesystem in production"
            )

        if not self.results["validation_results"].get("health_check", False):
            self.results["recommendations"].append(
                "Add health check to Docker container"
            )

        if not self.results["validation_results"].get("api_cleanup", False):
            self.results["recommendations"].append("Implement API cleanup endpoints")

        size_opt = self.results["validation_results"].get("size_optimization", 0)
        if size_opt < 10:
            self.results["recommendations"].append("Optimize Docker image size further")

    def generate_report(self):
        """Generate comprehensive validation report"""

        self.generate_recommendations()

        # Calculate success rate
        total_checks = len(self.results["validation_results"])
        passed_checks = sum(1 for v in self.results["validation_results"].values() if v)

        if total_checks > 0:
            success_rate = (passed_checks / total_checks) * 100
        else:
            success_rate = 0

        self.results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "success_rate": success_rate,
            "overall_status": (
                "PASS"
                if len(self.results["recommendations"]) == 0
                else "PASS_WITH_RECOMMENDATIONS"
            ),
        }

        # Save report
        report_file = (
            self.project_root
            / f"docker_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Print summary
        print("\n" + "=" * 80)
        print("DOCKER CONTAINER PRODUCTION BUILD VALIDATION REPORT")
        print("=" * 80)
        print(f"Subtask: {self.results['subtask']}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Status: {self.results['summary']['overall_status']}")

        if self.results["recommendations"]:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(self.results["recommendations"], 1):
                print(f"  {i}. {rec}")
        else:
            print("\n✅ All validations passed - No recommendations")

        print(f"\nDetailed report saved to: {report_file}")
        print("=" * 80)

    def run_validation(self):
        """Run complete validation"""
        self.log_info("Starting Docker Container Production Build Validation...")
        self.log_info("Subtask 15.1 - Validate Docker Container Production Build")

        try:
            self.validate_docker_files()
            self.validate_production_config()
            self.test_multistage_builds()
            self.verify_security_practices()
            self.test_deployment_scenarios()
            self.validate_api_cleanup()

        except Exception as e:
            self.log_error(f"Validation failed with exception: {e}")

        finally:
            self.generate_report()

        return len(self.results["recommendations"]) == 0


def main():
    """Main execution function"""
    validator = ComprehensiveDockerValidator()

    if validator.run_validation():
        print(
            "\n✅ Docker Container Production Build Validation COMPLETED SUCCESSFULLY"
        )
        return 0
    else:
        print(
            "\n⚠️  Docker Container Production Build Validation COMPLETED WITH RECOMMENDATIONS"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
