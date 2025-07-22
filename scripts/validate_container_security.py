#!/usr/bin/env python3
"""
Container Security Validation Script
Validates Docker container security best practices
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


class ContainerSecurityValidator:
    """
    Container security best practices validator
    """

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.getcwd())
        self.security_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "dockerfile_security": {},
            "image_security": {},
            "runtime_security": {},
            "vulnerabilities": {},
            "recommendations": [],
        }

    def log_info(self, message: str):
        """Log informational message"""
        print(f"[INFO] {message}")

    def log_warning(self, message: str):
        """Log warning message"""
        print(f"[WARNING] {message}")

    def log_error(self, message: str):
        """Log error message"""
        print(f"[ERROR] {message}")

    def run_command(self, command: Sequence[str], timeout: int = 300) -> Tuple[int, str, str]:
        """Run a command and return return code, stdout, stderr"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return 1, "", str(e)

    def validate_dockerfile_security(self) -> bool:
        """Validate Dockerfile security best practices"""
        self.log_info("Validating Dockerfile security practices...")

        dockerfile_path = self.project_root / "Dockerfile"
        if not dockerfile_path.exists():
            self.log_error("Dockerfile not found")
            return False

        try:
            with open(dockerfile_path, "r") as f:
                content = f.read()

            security_checks: Dict[str, Dict[str, Any]] = {
                "non_root_user": {
                    "pattern": "USER ",
                    "description": "Non-root user configured",
                    "required": True,
                },
                "health_check": {
                    "pattern": "HEALTHCHECK",
                    "description": "Health check configured",
                    "required": True,
                },
                "no_sudo": {
                    "pattern": "sudo",
                    "description": "No sudo usage",
                    "required": True,
                    "inverse": True,
                },
                "no_curl_pipe": {
                    "pattern": "curl.*|.*sh",
                    "description": "No curl pipe to shell",
                    "required": True,
                    "inverse": True,
                },
                "minimal_packages": {
                    "pattern": "rm -rf /var/lib/apt/lists",
                    "description": "Package cache cleanup",
                    "required": True,
                },
                "secure_permissions": {
                    "pattern": "chmod 600",
                    "description": "Secure file permissions",
                    "required": False,
                },
            }

            for check_name, check_config in security_checks.items():
                pattern: str = check_config["pattern"]
                description: str = check_config["description"]
                required: bool = check_config.get("required", False)
                inverse: bool = check_config.get("inverse", False)

                found = pattern in content
                if inverse:
                    found = not found

                self.security_results["dockerfile_security"][check_name] = {
                    "status": found,
                    "description": description,
                    "required": required,
                }

                if found:
                    self.log_info(f"✓ {description}")
                else:
                    if required:
                        self.log_error(f"✗ {description} - REQUIRED")
                    else:
                        self.log_warning(f"⚠ {description} - RECOMMENDED")

            return True

        except Exception as e:
            self.log_error(f"Error validating Dockerfile: {e}")
            return False

    def scan_image_vulnerabilities(self, image_name: str) -> bool:
        """Scan Docker image for vulnerabilities"""
        self.log_info(f"Scanning {image_name} for vulnerabilities...")

        # Try multiple vulnerability scanners
        scanners = [
            {
                "name": "trivy",
                "command": [
                    "trivy",
                    "image",
                    "--format",
                    "json",
                    "--quiet",
                    image_name,
                ],
            },
            {
                "name": "docker-scout",
                "command": ["docker", "scout", "cves", image_name],
            },
        ]

        vulnerability_found = False

        for scanner in scanners:
            self.log_info(f"Trying {scanner['name']} scanner...")

            # Check if scanner is available
            if scanner["name"] == "trivy":
                check_cmd = ["trivy", "--version"]
            else:
                check_cmd = ["docker", "scout", "version"]

            returncode, stdout, stderr = self.run_command(check_cmd)
            if returncode != 0:
                self.log_warning(f"{scanner['name']} not available")
                continue

            # Run scan
            returncode, stdout, stderr = self.run_command(scanner["command"])
            if returncode == 0:
                if scanner["name"] == "trivy":
                    try:
                        scan_result = json.loads(stdout)
                        if "Results" in scan_result:
                            total_vulns = 0
                            high_vulns = 0
                            critical_vulns = 0

                            for result in scan_result["Results"]:
                                if "Vulnerabilities" in result:
                                    for vuln in result["Vulnerabilities"]:
                                        total_vulns += 1
                                        severity = vuln.get("Severity", "").upper()
                                        if severity == "HIGH":
                                            high_vulns += 1
                                        elif severity == "CRITICAL":
                                            critical_vulns += 1

                            self.security_results["vulnerabilities"][scanner["name"]] = {
                                "total": total_vulns,
                                "high": high_vulns,
                                "critical": critical_vulns,
                            }

                            self.log_info(f"Vulnerabilities found: {total_vulns}")
                            self.log_info(f"High severity: {high_vulns}")
                            self.log_info(f"Critical severity: {critical_vulns}")

                            if critical_vulns > 0:
                                self.log_error(f"Critical vulnerabilities found: {critical_vulns}")
                                vulnerability_found = True
                            elif high_vulns > 0:
                                self.log_warning(
                                    f"High severity vulnerabilities found: {high_vulns}"
                                )
                                vulnerability_found = True
                            else:
                                self.log_info("✓ No high or critical vulnerabilities found")

                    except json.JSONDecodeError:
                        self.log_warning(f"Could not parse {scanner['name']} output")

                break  # Use first available scanner
            else:
                self.log_warning(f"{scanner['name']} scan failed: {stderr}")

        if not vulnerability_found:
            self.log_info("No vulnerability scanner found or no vulnerabilities detected")

        return True

    def test_runtime_security(self, image_name: str) -> bool:
        """Test runtime security configurations"""
        self.log_info("Testing runtime security configurations...")

        container_name = "security-test-container"

        # Clean up any existing container
        self.run_command(["docker", "rm", "-f", container_name])

        # Test 1: Read-only filesystem
        self.log_info("Testing read-only filesystem...")
        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "run",
                "--rm",
                "--read-only",
                "--tmpfs",
                "/tmp",  # nosec B108 - Secure tmpfs mount in Docker
                image_name,
                "touch",
                "/test-file",
            ]
        )

        if returncode != 0:
            self.log_info("✓ Read-only filesystem prevents file creation")
            self.security_results["runtime_security"]["readonly_fs"] = True
        else:
            self.log_warning("⚠ Read-only filesystem not enforced")
            self.security_results["runtime_security"]["readonly_fs"] = False

        # Test 2: Non-root user
        self.log_info("Testing non-root user...")
        returncode, stdout, stderr = self.run_command(
            ["docker", "run", "--rm", image_name, "whoami"]
        )

        if returncode == 0:
            user = stdout.strip()
            if user != "root":
                self.log_info(f"✓ Container runs as non-root user: {user}")
                self.security_results["runtime_security"]["non_root_user"] = True
            else:
                self.log_error("✗ Container runs as root user")
                self.security_results["runtime_security"]["non_root_user"] = False
        else:
            self.log_warning("Could not determine container user")
            self.security_results["runtime_security"]["non_root_user"] = False

        # Test 3: No new privileges
        self.log_info("Testing no new privileges...")
        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "run",
                "--rm",
                "--security-opt",
                "no-new-privileges:true",
                image_name,
                "echo",
                "test",
            ]
        )

        if returncode == 0:
            self.log_info("✓ No new privileges security option works")
            self.security_results["runtime_security"]["no_new_privileges"] = True
        else:
            self.log_warning("⚠ No new privileges security option failed")
            self.security_results["runtime_security"]["no_new_privileges"] = False

        # Test 4: Capabilities drop
        self.log_info("Testing capabilities drop...")
        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "run",
                "--rm",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "CHOWN",
                "--cap-add",
                "DAC_OVERRIDE",
                "--cap-add",
                "FOWNER",
                "--cap-add",
                "SETGID",
                "--cap-add",
                "SETUID",
                image_name,
                "echo",
                "test",
            ]
        )

        if returncode == 0:
            self.log_info("✓ Minimal capabilities configuration works")
            self.security_results["runtime_security"]["minimal_capabilities"] = True
        else:
            self.log_warning("⚠ Minimal capabilities configuration failed")
            self.security_results["runtime_security"]["minimal_capabilities"] = False

        return True

    def test_network_security(self, image_name: str) -> bool:
        """Test network security configurations"""
        self.log_info("Testing network security...")

        container_name = "network-security-test"

        # Clean up any existing container
        self.run_command(["docker", "rm", "-f", container_name])

        # Test isolated network
        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--network",
                "none",
                image_name,
            ]
        )

        if returncode == 0:
            self.log_info("✓ Container can run with isolated network")
            self.security_results["runtime_security"]["network_isolation"] = True
        else:
            self.log_warning("⚠ Container cannot run with isolated network")
            self.security_results["runtime_security"]["network_isolation"] = False

        # Clean up
        self.run_command(["docker", "rm", "-f", container_name])

        return True

    def analyze_image_layers(self, image_name: str) -> bool:
        """Analyze image layers for security concerns"""
        self.log_info("Analyzing image layers for security concerns...")

        # Get image history
        returncode, stdout, stderr = self.run_command(
            ["docker", "history", "--no-trunc", image_name]
        )

        if returncode == 0:
            lines = stdout.strip().split("\n")
            if len(lines) > 1:
                layer_count = len(lines) - 1
                self.log_info(f"Image has {layer_count} layers")

                # Check for secrets in layers
                secrets_found = False
                for line in lines:
                    if any(
                        keyword in line.lower()
                        for keyword in ["password", "secret", "key", "token"]
                    ):
                        secrets_found = True
                        break

                if secrets_found:
                    self.log_warning("⚠ Potential secrets found in image layers")
                    self.security_results["image_security"]["secrets_in_layers"] = True
                else:
                    self.log_info("✓ No obvious secrets found in image layers")
                    self.security_results["image_security"]["secrets_in_layers"] = False

                # Check layer size distribution
                large_layers = 0
                for line in lines[1:]:  # Skip header
                    if "MB" in line:
                        size_part = line.split()[-1]
                        if size_part.replace("MB", "").replace(".", "").isdigit():
                            size = float(size_part.replace("MB", ""))
                            if size > 100:  # Layers > 100MB
                                large_layers += 1

                if large_layers > 0:
                    self.log_warning(f"⚠ {large_layers} large layers (>100MB) found")
                    self.security_results["image_security"]["large_layers"] = large_layers
                else:
                    self.log_info("✓ No excessively large layers found")
                    self.security_results["image_security"]["large_layers"] = 0

        return True

    def generate_security_recommendations(self):
        """Generate security recommendations"""
        self.log_info("Generating security recommendations...")

        recommendations = []

        # Dockerfile security recommendations
        dockerfile_security = self.security_results.get("dockerfile_security", {})
        for check_name, check_result in dockerfile_security.items():
            if check_result["required"] and not check_result["status"]:
                recommendations.append(f"Fix Dockerfile security: {check_result['description']}")

        # Runtime security recommendations
        runtime_security = self.security_results.get("runtime_security", {})
        if not runtime_security.get("readonly_fs", False):
            recommendations.append("Enable read-only filesystem with --read-only flag")

        if not runtime_security.get("non_root_user", False):
            recommendations.append("Configure container to run as non-root user")

        if not runtime_security.get("no_new_privileges", False):
            recommendations.append("Add --security-opt no-new-privileges:true")

        if not runtime_security.get("minimal_capabilities", False):
            recommendations.append("Drop all capabilities and add only necessary ones")

        # Vulnerability recommendations
        vulnerabilities = self.security_results.get("vulnerabilities", {})
        for scanner, vuln_data in vulnerabilities.items():
            if vuln_data.get("critical", 0) > 0:
                recommendations.append(f"Fix {vuln_data['critical']} critical vulnerabilities")
            if vuln_data.get("high", 0) > 0:
                recommendations.append(f"Fix {vuln_data['high']} high severity vulnerabilities")

        # Image security recommendations
        image_security = self.security_results.get("image_security", {})
        if image_security.get("secrets_in_layers", False):
            recommendations.append("Remove secrets from image layers")

        if image_security.get("large_layers", 0) > 0:
            recommendations.append("Optimize large layers to reduce attack surface")

        self.security_results["recommendations"] = recommendations

        return recommendations

    def generate_report(self):
        """Generate comprehensive security report"""
        self.log_info("Generating security report...")

        recommendations = self.generate_security_recommendations()

        # Calculate security score
        total_checks = 0
        passed_checks = 0

        for category in [
            "dockerfile_security",
            "runtime_security",
            "image_security",
        ]:
            if category in self.security_results:
                for check_name, check_result in self.security_results[category].items():
                    if isinstance(check_result, dict) and "status" in check_result:
                        total_checks += 1
                        if check_result["status"]:
                            passed_checks += 1
                    elif isinstance(check_result, bool):
                        total_checks += 1
                        if check_result:
                            passed_checks += 1

        security_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0

        self.security_results["summary"] = {
            "security_score": security_score,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "recommendations_count": len(recommendations),
        }

        # Save report
        report_file = (
            self.project_root
            / f"container_security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(self.security_results, f, indent=2)

        self.log_info(f"Security report saved to: {report_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("CONTAINER SECURITY VALIDATION REPORT")
        print("=" * 80)
        print(f"Security Score: {security_score:.1f}%")
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Recommendations: {len(recommendations)}")

        if recommendations:
            print("\nSECURITY RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("\n✅ No security recommendations - Good job!")

        print("=" * 80)

    def validate_all(self, image_name: str = "freeagentics:production"):
        """Run all security validation checks"""
        self.log_info("Starting container security validation...")

        # Build image if it doesn't exist
        returncode, stdout, stderr = self.run_command(["docker", "inspect", image_name])

        if returncode != 0:
            self.log_info("Building production image for security testing...")
            returncode, stdout, stderr = self.run_command(
                [
                    "docker",
                    "build",
                    "--target",
                    "production",
                    "-t",
                    image_name,
                    ".",
                ]
            )

            if returncode != 0:
                self.log_error(f"Failed to build image: {stderr}")
                return False

        try:
            self.validate_dockerfile_security()
            self.scan_image_vulnerabilities(image_name)
            self.test_runtime_security(image_name)
            self.test_network_security(image_name)
            self.analyze_image_layers(image_name)

        except Exception as e:
            self.log_error(f"Security validation failed: {e}")

        finally:
            self.generate_report()

        # Clean up test image
        self.run_command(["docker", "rmi", image_name])

        return len(self.security_results.get("recommendations", [])) == 0


def main():
    """Main execution function"""
    validator = ContainerSecurityValidator()

    if len(sys.argv) > 1:
        image_name = sys.argv[1]
    else:
        image_name = "freeagentics:production"

    if validator.validate_all(image_name):
        print("\n✅ Container security validation PASSED")
        return 0
    else:
        print("\n⚠ Container security validation completed with recommendations")
        return 1


if __name__ == "__main__":
    sys.exit(main())
