#!/usr/bin/env python3
"""
Test script for Deployment Validation

Tests the export validation, hardware compatibility, and deployment verification
modules.
"""

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

from deployment.deployment_verification import DeploymentVerifier
from deployment.export_validator import ExportValidator, HardwarePlatform
from deployment.hardware_compatibility import CompatibilityTester

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_package(package_dir: Path):
    """Create a test deployment package"""
    # Create directory structure
    package_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest = {
        "package_name": "test_agent",
        "version": "1.0.0",
        "agent_class": "explorer",
        "created_at": "2024-01-01T00:00:00Z",
        "platform": "generic",
        "dependencies": ["numpy", "requests"],
        "files": {},
        "metadata": {"description": "Test agent package"},
    }

    # Create required files
    files = {
        "manifest.json": manifest,
        "agent_config.json": {
            "agent_class": "explorer",
            "personality": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.6,
                "agreeableness": 0.8,
                "neuroticism": 0.3,
            },
        },
        "gnn_model.json": {
            "metadata": {"agent_class": "explorer", "version": "1.0"},
            "layers": [
                {"type": "input", "size": 10},
                {"type": "hidden", "size": 20},
                {"type": "output", "size": 5},
            ],
        },
        "requirements.txt": "numpy>=1.20.0\nrequests>=2.25.0\naiohttp>=3.8.0",
        "README.md": "# Test Agent\n\nThis is a test agent package.",
        "run.sh": """#!/bin/bash
echo "Starting test agent..."
python -m agent.main
""",
        "health_check.py": """#!/usr/bin/env python3
import sys
print("Agent is healthy")
sys.exit(0)
""",
        "deployment_config.json": {
            "agent_name": "test_agent",
            "base_url": "http://localhost:8080",
            "health_checks": [
                {
                    "name": "basic",
                    "endpoint": "/health",
                    "method": "GET",
                    "expected_status": 200,
                }
            ],
        },
    }

    # Write files and calculate hashes
    file_hashes = {}
    for filename, content in files.items():
        file_path = package_dir / filename

        if isinstance(content, dict):
            content_str = json.dumps(content, indent=2)
        else:
            content_str = content

        file_path.write_text(content_str)

        # Calculate hash
        import hashlib

        file_hash = hashlib.sha256(content_str.encode()).hexdigest()
        file_hashes[filename] = file_hash

    # Update manifest with file hashes
    manifest["files"] = file_hashes
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Make scripts executable
    (package_dir / "run.sh").chmod(0o755)
    (package_dir / "health_check.py").chmod(0o755)

    # Create logs directory
    (package_dir / "logs").mkdir(exist_ok=True)

    logger.info(f"Created test package at: {package_dir}")


def test_export_validation() -> None:
    """Test export validation"""
    print("\n=== Testing Export Validation ===")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        package_dir = Path(temp_dir) / "test_package"
        create_test_package(package_dir)

        # Test validation
        validator = ExportValidator()

        # Test without platform
        print("\n1. Testing basic validation...")
        results = validator.validate_package(package_dir)

        # Print results
        for result in results:
            print(f"  {result.check_name}: {result.status.value} - {result.message}")

        # Test with specific platform
        print("\n2. Testing platform-specific validation...")
        platforms = [
            HardwarePlatform.RASPBERRY_PI,
            HardwarePlatform.MAC_MINI,
            HardwarePlatform.JETSON_NANO,
        ]

        for platform in platforms:
            print(f"\n  Platform: {platform.value}")
            results = validator.validate_package(package_dir, platform)

            # Show only platform-specific results
            platform_results = [
                r for r in results if "platform" in r.check_name or "hardware" in r.check_name
            ]

            for result in platform_results:
                print(f"    {result.check_name}: {result.status.value}")

        # Test with missing files
        print("\n3. Testing with missing files...")
        (package_dir / "run.sh").unlink()

        results = validator.validate_package(package_dir)
        failed_checks = [r for r in results if r.status.value == "failed"]

        print(f"  Found {len(failed_checks)} failed checks")
        for result in failed_checks[:3]:
            print(f"    {result.check_name}: {result.message}")


def test_hardware_compatibility() -> None:
    """Test hardware compatibility"""
    print("\n=== Testing Hardware Compatibility ===")

    # Create test package
    with tempfile.TemporaryDirectory() as temp_dir:
        package_dir = Path(temp_dir) / "test_package"
        create_test_package(package_dir)

        # Run compatibility tests
        tester = CompatibilityTester()

        print("\n1. Detecting current hardware...")
        hardware = tester.detector.detect_hardware()

        print(f"  Platform: {hardware.name}")
        print(f"  Architecture: {hardware.architecture}")
        print(f"  CPU: {hardware.cpu_model}")
        print(f"  Cores: {hardware.cpu_cores}")
        print(f"  RAM: {hardware.ram_gb}GB")
        print(f"  GPU: {hardware.gpu_model or 'None'}")

        print("\n2. Running compatibility tests...")
        results = tester.run_tests(package_dir, hardware)

        # Show test results
        for test_result in results["test_results"]:
            status = test_result["status"]
            icon = {"passed": "✓", "failed": "✗", "skipped": "○", "timeout": "⏱"}.get(status, "?")

            print(f"  {icon} {test_result['test_name']}: {test_result['message']}")

        # Show summary
        summary = results["summary"]
        print(f"\n  Summary: {summary['passed']}/{summary['total_tests']} passed")
        print(f"  Duration: {summary['duration']:.1f}s")


async def test_deployment_verification():
    """Test deployment verification"""
    print("\n=== Testing Deployment Verification ===")

    # Create test deployment
    with tempfile.TemporaryDirectory() as temp_dir:
        deployment_dir = Path(temp_dir) / "deployment"
        create_test_package(deployment_dir)

        # Create mock PID file (simulate running service)
        import os

        pid_file = deployment_dir / "agent.pid"
        pid_file.write_text(str(os.getpid()))  # Use current process PID

        # Create some log entries
        log_file = deployment_dir / "agent.log"
        log_file.write_text(
            """
2024-01-01 10:00:00 INFO Starting agent...
2024-01-01 10:00:01 INFO Agent initialized
2024-01-01 10:00:02 WARNING Low memory detected
2024-01-01 10:00:03 INFO Agent ready
2024-01-01 10:00:04 ERROR Failed to connect to peer
2024-01-01 10:00:05 INFO Retrying connection...
"""
        )

        # Run verification
        verifier = DeploymentVerifier(deployment_dir)

        print("\n1. Checking service status...")
        service_running = verifier.service_manager.is_running()
        print(f"  Service running: {service_running}")

        if service_running:
            metrics = verifier.service_manager.get_service_metrics()
            print(f"  PID: {metrics.get('pid')}")
            print(f"  Memory: {metrics.get('memory_mb', 0):.1f}MB")
            print(f"  CPU: {metrics.get('cpu_percent', 0):.1f}%")

        print("\n2. Running deployment verification...")
        results = await verifier.verify_deployment()

        # Show results
        print(f"\n  Overall Status: {results['overall_status']}")

        for check_name, check_results in results["checks"].items():
            print(f"\n  {check_name.title()} Check:")
            if isinstance(check_results, dict):
                for key, value in check_results.items():
                    if key != "metrics" and key != "results":
                        print(f"    {key}: {value}")


async def test_health_monitoring():
    """Test health monitoring"""
    print("\n=== Testing Health Monitoring ===")

    from deployment.deployment_verification import DeploymentConfig, HealthCheck, HealthMonitor

    # Create test configuration
    config = DeploymentConfig(
        agent_name="test_agent",
        base_url="http://localhost:8080",
        health_checks=[
            HealthCheck(
                name="basic",
                endpoint="/health",
                method="GET",
                expected_status=200,
                timeout=5,
            )
        ],
        pid_file=Path("/tmp/agent.pid"),
        log_file=Path("/tmp/agent.log"),
    )

    monitor = HealthMonitor(config)

    print("\n1. Performing health check...")
    # This will fail since there's no actual service running
    try:
        status = await monitor.check_health()
        print(f"  Health status: {status.value}")
    except Exception as e:
        print(f"  Health check failed (expected): {e}")

    print("\n2. Getting health metrics...")
    metrics = monitor.get_health_metrics()
    print(f"  Status: {metrics['status']}")
    print(f"  Uptime: {metrics['uptime_percent']:.1f}%")


def test_package_compression() -> None:
    """Test package compression and extraction"""
    print("\n=== Testing Package Compression ===")

    import zipfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test package
        package_dir = Path(temp_dir) / "test_package"
        create_test_package(package_dir)

        # Compress to ZIP
        zip_path = Path(temp_dir) / "test_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in package_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir.parent)
                    zf.write(file_path, arcname)

        print(f"  Created ZIP package: {zip_path}")
        print(f"  Size: {zip_path.stat().st_size / 1024:.1f}KB")

        # Test validation on compressed package
        validator = ExportValidator()
        results = validator.validate_package(zip_path)

        # Check summary
        summary = next((r for r in results if r.check_name == "validation_summary"), None)
        if summary:
            print(f"  Validation: {summary.status.value}")
            print(f"  {summary.message}")


async def main():
    """Run all tests"""
    print("=== Deployment Validation Test Suite ===")

    # Test export validation
    test_export_validation()

    # Test hardware compatibility
    test_hardware_compatibility()

    # Test deployment verification
    await test_deployment_verification()

    # Test health monitoring
    await test_health_monitoring()

    # Test package compression
    test_package_compression()

    print("\n=== All Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
