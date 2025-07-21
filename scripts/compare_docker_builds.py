#!/usr/bin/env python3
"""
Compare Docker Build Sizes and Performance.
"""

import subprocess
import sys
import time


def run_command(command, timeout=300):
    """Run a command with timeout."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return 1, "", str(e)


def get_image_size(image_name):
    """Get image size in bytes."""
    returncode, stdout, stderr = run_command(
        ["docker", "inspect", image_name, "--format", "{{.Size}}"]
    )

    if returncode == 0:
        return int(stdout.strip())
    return 0


def get_human_readable_size(size_bytes):
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    """Compare Docker builds."""
    print("=== Docker Build Size Comparison ===")

    builds = [
        {
            "name": "Original",
            "dockerfile": "Dockerfile",
            "tag": "freeagentics:original",
        },
        {
            "name": "Optimized",
            "dockerfile": "Dockerfile.optimized",
            "tag": "freeagentics:optimized",
        },
    ]

    results = {}

    for build in builds:
        print(f"\n--- Building {build['name']} Version ---")

        # Build image
        start_time = time.time()
        returncode, stdout, stderr = run_command(
            [
                "docker",
                "build",
                "--target",
                "production",
                "-f",
                build["dockerfile"],
                "-t",
                build["tag"],
                ".",
            ]
        )
        build_time = time.time() - start_time

        if returncode == 0:
            print(f"✅ {build['name']} build successful in {build_time:.2f}s")

            # Get image size
            size_bytes = get_image_size(build["tag"])
            size_human = get_human_readable_size(size_bytes)

            results[build["name"]] = {
                "success": True,
                "build_time": build_time,
                "size_bytes": size_bytes,
                "size_human": size_human,
                "tag": build["tag"],
            }

            print(f"📏 {build['name']} size: {size_human}")

        else:
            print(f"❌ {build['name']} build failed: {stderr}")
            results[build["name"]] = {"success": False, "error": stderr}

    # Compare results
    print("\n=== Comparison Results ===")

    if all(r.get("success", False) for r in results.values()):
        original_size = results["Original"]["size_bytes"]
        optimized_size = results["Optimized"]["size_bytes"]

        if original_size > 0 and optimized_size > 0:
            reduction = (original_size - optimized_size) / original_size * 100

            print(f"Original size: {results['Original']['size_human']}")
            print(f"Optimized size: {results['Optimized']['size_human']}")
            print(f"Size reduction: {reduction:.1f}%")

            if reduction > 0:
                print("✅ Optimization successful!")
            else:
                print("⚠️  Optimization didn't reduce size")

        # Test both containers
        print("\n--- Testing Container Functionality ---")

        for build_name, result in results.items():
            if result.get("success"):
                print(f"Testing {build_name} container...")

                container_name = f"test-{build_name.lower()}"

                # Clean up any existing container
                run_command(["docker", "rm", "-f", container_name])

                # Run container
                returncode, stdout, stderr = run_command(
                    [
                        "docker",
                        "run",
                        "-d",
                        "--name",
                        container_name,
                        "-e",
                        "ENVIRONMENT=test",
                        result["tag"],
                    ]
                )

                if returncode == 0:
                    time.sleep(5)  # Wait for startup

                    # Check if container is running
                    returncode, stdout, stderr = run_command(
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
                        print(f"✅ {build_name} container running")
                    else:
                        print(f"❌ {build_name} container failed to start")

                    # Clean up
                    run_command(["docker", "rm", "-f", container_name])
                else:
                    print(f"❌ {build_name} container failed to run: {stderr}")

    # Layer analysis
    print("\n--- Layer Analysis ---")

    for build_name, result in results.items():
        if result.get("success"):
            print(f"\n{build_name} layers:")

            returncode, stdout, stderr = run_command(
                [
                    "docker",
                    "history",
                    result["tag"],
                    "--format",
                    "table {{.CreatedBy}}\t{{.Size}}",
                ]
            )

            if returncode == 0:
                lines = stdout.strip().split("\n")
                if len(lines) > 1:
                    print(f"  Total layers: {len(lines) - 1}")

                    # Count non-zero layers
                    non_zero_layers = 0
                    for line in lines[1:]:
                        if "B" in line and not line.strip().endswith("0B"):
                            non_zero_layers += 1

                    print(f"  Non-zero layers: {non_zero_layers}")

    # Clean up test images
    print("\n--- Cleaning up test images ---")
    for build_name, result in results.items():
        if result.get("success"):
            print(f"Removing {result['tag']}...")
            run_command(["docker", "rmi", result["tag"]])

    print("\n=== Comparison Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
