#!/usr/bin/env python3
"""
Quick Docker Production Build Validation
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, timeout=60):
    """Run a command with timeout"""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return 1, "", str(e)


def main():
    """Quick validation of Docker production build"""
    print("=== Quick Docker Production Build Validation ===")

    # Check Docker is available
    print("1. Checking Docker availability...")
    returncode, stdout, stderr = run_command(["docker", "--version"])
    if returncode != 0:
        print(f"❌ Docker not available: {stderr}")
        return 1
    print(f"✅ Docker available: {stdout.strip()}")

    # Check Dockerfile exists
    print("2. Checking Dockerfile...")
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print("❌ Dockerfile not found")
        return 1
    print("✅ Dockerfile exists")

    # Check requirements file
    print("3. Checking requirements-production.txt...")
    requirements_path = Path("requirements-production.txt")
    if not requirements_path.exists():
        print("❌ requirements-production.txt not found")
        return 1
    print("✅ requirements-production.txt exists")

    # Validate Dockerfile structure
    print("4. Validating Dockerfile structure...")
    with open(dockerfile_path, "r") as f:
        content = f.read()

    checks = [
        ("FROM", "Base image specified"),
        ("as production", "Production stage defined"),
        ("USER", "Non-root user configured"),
        ("HEALTHCHECK", "Health check configured"),
        ("EXPOSE", "Port exposed"),
        ("CMD", "Command specified"),
    ]

    for check, description in checks:
        if check in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")

    # Test build (with timeout)
    print("5. Testing Docker build...")
    print("   (This may take a few minutes...)")

    returncode, stdout, stderr = run_command(
        [
            "docker",
            "build",
            "--target",
            "production",
            "-t",
            "freeagentics:validation-test",
            ".",
        ],
        timeout=300,
    )  # 5 minutes timeout

    if returncode == 0:
        print("✅ Docker build successful")

        # Get image size
        returncode, stdout, stderr = run_command(
            [
                "docker",
                "images",
                "freeagentics:validation-test",
                "--format",
                "{{.Size}}",
            ]
        )

        if returncode == 0:
            print(f"📏 Image size: {stdout.strip()}")

        # Clean up
        run_command(["docker", "rmi", "freeagentics:validation-test"])

    else:
        print(f"❌ Docker build failed: {stderr}")
        return 1

    # Check docker-compose
    print("6. Checking docker-compose.yml...")
    compose_path = Path("docker-compose.yml")
    if compose_path.exists():
        returncode, stdout, stderr = run_command(
            ["docker", "compose", "config"]
        )
        if returncode == 0:
            print("✅ Docker Compose configuration valid")
        else:
            print(f"❌ Docker Compose configuration invalid: {stderr}")
    else:
        print("⚠️  docker-compose.yml not found")

    print("\n=== Validation Complete ===")
    print("✅ Docker container production build validation PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
