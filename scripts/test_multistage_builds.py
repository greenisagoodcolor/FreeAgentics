#!/usr/bin/env python3
"""
Multi-stage Docker Build Testing Script
Tests and validates multi-stage Docker builds and optimization
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


class MultiStageBuilder:
    """
    Multi-stage Docker build tester and optimizer
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.build_results: Dict[str, Any] = {}

    def log_info(self, message: str):
        """Log informational message"""
        print(f"[INFO] {message}")

    def log_error(self, message: str):
        """Log error message"""
        print(f"[ERROR] {message}")

    def run_command(self, command: List[str], timeout: int = 600) -> Tuple[int, str, str]:
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

    def get_image_size(self, image_name: str) -> int:
        """Get image size in bytes"""
        returncode, stdout, stderr = self.run_command(
            ["docker", "inspect", image_name, "--format", "{{.Size}}"]
        )

        if returncode == 0:
            return int(stdout.strip())
        return 0

    def build_stage(self, stage_name: str, image_tag: str) -> bool:
        """Build a specific stage"""
        self.log_info(f"Building {stage_name} stage...")

        build_command = [
            "docker",
            "build",
            "--target",
            stage_name,
            "-t",
            image_tag,
            ".",
        ]

        start_time = time.time()
        returncode, stdout, stderr = self.run_command(build_command)
        build_time = time.time() - start_time

        if returncode == 0:
            image_size = self.get_image_size(image_tag)
            self.build_results[stage_name] = {
                "success": True,
                "build_time": build_time,
                "image_size": image_size,
                "image_tag": image_tag,
            }
            self.log_info(f"✓ {stage_name} stage built successfully in {build_time:.2f}s")
            self.log_info(f"  Size: {image_size / (1024 * 1024):.1f} MB")
            return True
        else:
            self.log_error(f"✗ {stage_name} stage build failed: {stderr}")
            self.build_results[stage_name] = {
                "success": False,
                "build_time": build_time,
                "error": stderr,
            }
            return False

    def test_layer_caching(self):
        """Test Docker layer caching efficiency"""
        self.log_info("Testing layer caching efficiency...")

        # First build
        self.log_info("First build (cold cache)...")
        start_time = time.time()
        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "build",
                "--no-cache",
                "--target",
                "production",
                "-t",
                "freeagentics:cache-test-1",
                ".",
            ]
        )
        cold_build_time = time.time() - start_time

        if returncode != 0:
            self.log_error(f"Cold build failed: {stderr}")
            return

        # Second build (warm cache)
        self.log_info("Second build (warm cache)...")
        start_time = time.time()
        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "build",
                "--target",
                "production",
                "-t",
                "freeagentics:cache-test-2",
                ".",
            ]
        )
        warm_build_time = time.time() - start_time

        if returncode == 0:
            cache_efficiency = (cold_build_time - warm_build_time) / cold_build_time * 100
            self.log_info(f"Cold build time: {cold_build_time:.2f}s")
            self.log_info(f"Warm build time: {warm_build_time:.2f}s")
            self.log_info(f"Cache efficiency: {cache_efficiency:.1f}%")

            self.build_results["caching"] = {
                "cold_build_time": cold_build_time,
                "warm_build_time": warm_build_time,
                "cache_efficiency": cache_efficiency,
            }
        else:
            self.log_error(f"Warm build failed: {stderr}")

        # Clean up
        self.run_command(["docker", "rmi", "freeagentics:cache-test-1"])
        self.run_command(["docker", "rmi", "freeagentics:cache-test-2"])

    def analyze_layer_structure(self, image_tag: str):
        """Analyze Docker image layer structure"""
        self.log_info(f"Analyzing layer structure for {image_tag}...")

        # Get image history
        returncode, stdout, stderr = self.run_command(
            ["docker", "history", image_tag, "--no-trunc", "--format", "table"]
        )

        if returncode == 0:
            lines = stdout.strip().split("\n")
            if len(lines) > 1:
                self.log_info(f"Image has {len(lines) - 1} layers")

                # Count non-zero size layers
                non_zero_layers = 0
                for line in lines[1:]:  # Skip header
                    if "B" in line and not line.strip().endswith("0B"):
                        non_zero_layers += 1

                self.log_info(f"Non-zero size layers: {non_zero_layers}")
        else:
            self.log_error(f"Failed to get image history: {stderr}")

    def test_build_optimization(self):
        """Test various build optimization techniques"""
        self.log_info("Testing build optimization techniques...")

        # Test base image vs development vs production
        stages = [
            ("base", "freeagentics:base-test"),
            ("development", "freeagentics:dev-test"),
            ("production", "freeagentics:prod-test"),
        ]

        for stage_name, image_tag in stages:
            if self.build_stage(stage_name, image_tag):
                self.analyze_layer_structure(image_tag)

        # Compare sizes
        if "development" in self.build_results and "production" in self.build_results:
            dev_size = self.build_results["development"]["image_size"]
            prod_size = self.build_results["production"]["image_size"]

            if dev_size > 0 and prod_size > 0:
                reduction = (dev_size - prod_size) / dev_size * 100
                self.log_info(f"Size reduction from dev to prod: {reduction:.1f}%")

                if reduction > 0:
                    self.log_info("✓ Production stage provides size optimization")
                else:
                    self.log_info("⚠ Production stage doesn't reduce size")

    def test_security_hardening(self):
        """Test security hardening in multi-stage builds"""
        self.log_info("Testing security hardening...")

        # Test that production image runs as non-root
        returncode, stdout, stderr = self.run_command(
            ["docker", "run", "--rm", "freeagentics:prod-test", "whoami"]
        )

        if returncode == 0:
            user = stdout.strip()
            if user != "root":
                self.log_info(f"✓ Production image runs as non-root user: {user}")
            else:
                self.log_info("⚠ Production image runs as root user")
        else:
            self.log_error(f"Failed to check user: {stderr}")

    def test_production_readiness(self):
        """Test production readiness features"""
        self.log_info("Testing production readiness features...")

        # Test health check
        container_name = "multistage-health-test"

        # Clean up any existing container
        self.run_command(["docker", "rm", "-f", container_name])

        # Run container
        returncode, stdout, stderr = self.run_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "freeagentics:prod-test",
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
                    self.log_info("✓ Production image passes health check")
                else:
                    self.log_info(f"⚠ Health check status: {health_status}")
            else:
                self.log_info("⚠ No health check configured")
        else:
            self.log_error(f"Failed to run container: {stderr}")

        # Clean up
        self.run_command(["docker", "rm", "-f", container_name])

    def cleanup_test_images(self):
        """Clean up test images"""
        self.log_info("Cleaning up test images...")

        test_images = [
            "freeagentics:base-test",
            "freeagentics:dev-test",
            "freeagentics:prod-test",
        ]

        for image in test_images:
            self.run_command(["docker", "rmi", image])

    def generate_report(self):
        """Generate optimization report"""
        self.log_info("Generating multi-stage build report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "build_results": self.build_results,
            "recommendations": [],
        }

        # Add recommendations based on results
        if "development" in self.build_results and "production" in self.build_results:
            dev_size = self.build_results["development"]["image_size"]
            prod_size = self.build_results["production"]["image_size"]

            if dev_size > 0 and prod_size > 0:
                reduction = (dev_size - prod_size) / dev_size * 100
                if reduction < 10:
                    report["recommendations"].append(
                        "Consider further optimization of production stage"
                    )

        if "caching" in self.build_results:
            cache_efficiency = self.build_results["caching"]["cache_efficiency"]
            if cache_efficiency < 50:
                report["recommendations"].append(
                    "Consider optimizing Dockerfile for better layer caching"
                )

        # Save report
        report_file = (
            self.project_root
            / f"multistage_build_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.log_info(f"Multi-stage build report saved to: {report_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("MULTI-STAGE BUILD OPTIMIZATION REPORT")
        print("=" * 80)

        for stage_name, result in self.build_results.items():
            if stage_name == "caching":
                continue

            if result.get("success"):
                size_mb = result["image_size"] / (1024 * 1024)
                build_time = result["build_time"]
                print(f"{stage_name.upper()}: {size_mb:.1f} MB (built in {build_time:.2f}s)")
            else:
                print(f"{stage_name.upper()}: BUILD FAILED")

        if "caching" in self.build_results:
            cache_data = self.build_results["caching"]
            print(f"CACHE EFFICIENCY: {cache_data['cache_efficiency']:.1f}%")

        if report["recommendations"]:
            print("\nRECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")

        print("=" * 80)

    def run_all_tests(self):
        """Run all multi-stage build tests"""
        self.log_info("Starting multi-stage build optimization tests...")

        try:
            self.test_build_optimization()
            self.test_layer_caching()
            self.test_security_hardening()
            self.test_production_readiness()

        except Exception as e:
            self.log_error(f"Test failed with exception: {e}")

        finally:
            self.cleanup_test_images()
            self.generate_report()


def main():
    """Main execution function"""
    builder = MultiStageBuilder()
    builder.run_all_tests()

    return 0


if __name__ == "__main__":
    sys.exit(main())
