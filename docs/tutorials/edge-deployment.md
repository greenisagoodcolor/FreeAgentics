# Tutorial: Deploying Agent Coalitions to Edge Devices

This tutorial will guide you through the process of deploying agent coalitions to edge devices in the FreeAgentics system. Edge deployment allows coalitions to operate autonomously in the real world, independent of a central server.

## Prerequisites

Before starting this tutorial, make sure you have:

- Completed the [Creating an Agent tutorial](creating-an-agent.md)
- Completed the [Forming Agent Coalitions tutorial](coalition-formation.md)
- A functioning coalition ready for deployment
- An edge device (Raspberry Pi, Jetson Nano, or similar) with Docker installed
- Basic understanding of Docker and containerization

## Step 1: Import Required Modules

First, let's import the necessary modules for edge deployment.

```python
from coalitions.deployment.edge_packager import EdgePackager
from coalitions.deployment.deployment_manifest import DeploymentManifest
from infrastructure.export.coalition_packaging import CoalitionPackager
from infrastructure.deployment.hardware_compatibility import HardwareCompatibilityChecker
from infrastructure.deployment.deployment_verification import DeploymentVerifier
```

## Step 2: Check Coalition Readiness

Before deploying a coalition, you need to ensure it's ready for deployment.

```python
from readiness.readiness_evaluator import ReadinessEvaluator

# Create a readiness evaluator
readiness_evaluator = ReadinessEvaluator()

# Check if the coalition is ready for deployment
readiness_result = readiness_evaluator.evaluate_coalition(coalition)

if readiness_result.is_ready:
    print(f"Coalition '{coalition.name}' is ready for deployment")
    print(f"Readiness score: {readiness_result.score}/100")
    print(f"Strengths: {readiness_result.strengths}")
else:
    print(f"Coalition '{coalition.name}' is not ready for deployment")
    print(f"Readiness score: {readiness_result.score}/100")
    print(f"Issues to address: {readiness_result.issues}")
    # Exit if the coalition is not ready
    import sys
    sys.exit(1)
```

## Step 3: Check Hardware Compatibility

Ensure that the target edge device is compatible with the coalition's requirements.

```python
# Define the target device
target_device = {
    "type": "raspberry_pi_4b",
    "cpu": "ARM Cortex-A72",
    "ram": 4,  # GB
    "storage": 32,  # GB
    "gpu": None,
    "network": ["wifi", "ethernet"],
    "sensors": ["camera", "microphone"]
}

# Check hardware compatibility
compatibility_checker = HardwareCompatibilityChecker()
compatibility_result = compatibility_checker.check_compatibility(coalition, target_device)

if compatibility_result.is_compatible:
    print(f"Coalition '{coalition.name}' is compatible with the target device")
    print(f"Compatibility score: {compatibility_result.score}/100")
else:
    print(f"Coalition '{coalition.name}' is not compatible with the target device")
    print(f"Compatibility score: {compatibility_result.score}/100")
    print(f"Compatibility issues: {compatibility_result.issues}")
    # Exit if not compatible
    import sys
    sys.exit(1)
```

## Step 4: Create a Deployment Manifest

A deployment manifest defines the configuration and requirements for the coalition on the edge device.

```python
# Create a deployment manifest
manifest = DeploymentManifest(
    coalition_id=coalition.id,
    coalition_name=coalition.name,
    version="1.0.0",
    description="Resource gathering coalition for edge deployment",
    required_capabilities={
        "cpu": "ARM Cortex-A72",
        "ram_min": 2,  # GB
        "storage_min": 16,  # GB
        "network": ["wifi"],
        "sensors": ["camera"]
    },
    environment_variables={
        "LOG_LEVEL": "INFO",
        "MAX_ENERGY_CONSUMPTION": "50",
        "DATA_COLLECTION_INTERVAL": "300"
    },
    persistence={
        "enabled": True,
        "backup_interval": 3600,  # seconds
        "max_storage": 1  # GB
    },
    security={
        "encryption": "AES-256",
        "secure_boot": True,
        "access_control": "role_based"
    }
)
```

## Step 5: Package the Coalition

Now, let's package the coalition for deployment to the edge device.

```python
# Create a coalition packager
packager = CoalitionPackager()

# Package the coalition
package_result = packager.package_coalition(
    coalition=coalition,
    manifest=manifest,
    output_dir="./output",
    target_platform=target_device["type"]
)

if package_result.success:
    print(f"Coalition '{coalition.name}' packaged successfully")
    print(f"Package location: {package_result.package_path}")
    print(f"Package size: {package_result.package_size} MB")
else:
    print(f"Failed to package coalition '{coalition.name}'")
    print(f"Error: {package_result.error}")
    # Exit if packaging failed
    import sys
    sys.exit(1)
```

## Step 6: Create Docker Container

Create a Docker container for the coalition that can run on the edge device.

```python
# Create an edge packager
edge_packager = EdgePackager()

# Create a Docker container
container_result = edge_packager.create_container(
    package_path=package_result.package_path,
    manifest=manifest,
    target_platform=target_device["type"],
    output_dir="./containers"
)

if container_result.success:
    print(f"Docker container created successfully")
    print(f"Container image: {container_result.image_name}")
    print(f"Dockerfile location: {container_result.dockerfile_path}")
else:
    print(f"Failed to create Docker container")
    print(f"Error: {container_result.error}")
    # Exit if container creation failed
    import sys
    sys.exit(1)
```

## Step 7: Test the Container Locally

Before deploying to the edge device, it's a good practice to test the container locally.

```python
import subprocess

# Run the container locally
print("Testing container locally...")
try:
    result = subprocess.run(
        ["docker", "run", "--rm", "-p", "8080:8080", container_result.image_name],
        capture_output=True,
        text=True,
        timeout=30  # Run for 30 seconds for testing
    )
    print("Container test output:")
    print(result.stdout)
    if result.returncode != 0:
        print(f"Container test failed with return code {result.returncode}")
        print(f"Error: {result.stderr}")
        # Exit if test failed
        import sys
        sys.exit(1)
    else:
        print("Container test successful")
except Exception as e:
    print(f"Error testing container: {str(e)}")
    # Exit if test failed
    import sys
    sys.exit(1)
```

## Step 8: Push Container to Registry

Push the container to a registry where the edge device can access it.

```python
# Push the container to a registry
print("Pushing container to registry...")
try:
    # Tag the image for the registry
    registry_url = "registry.example.com"
    registry_image = f"{registry_url}/{container_result.image_name}"

    subprocess.run(
        ["docker", "tag", container_result.image_name, registry_image],
        check=True
    )

    # Push the image
    subprocess.run(
        ["docker", "push", registry_image],
        check=True
    )

    print(f"Container pushed to registry: {registry_image}")
except Exception as e:
    print(f"Error pushing container to registry: {str(e)}")
    # Exit if push failed
    import sys
    sys.exit(1)
```

## Step 9: Deploy to Edge Device

Deploy the container to the edge device.

```python
from infrastructure.deployment.edge_deployer import EdgeDeployer

# Create an edge deployer
edge_deployer = EdgeDeployer()

# Define the edge device connection details
edge_device = {
    "name": "raspberry-pi-01",
    "host": "192.168.1.100",
    "port": 22,
    "username": "pi",
    "key_file": "~/.ssh/id_rsa",
    "platform": "raspberry_pi_4b"
}

# Deploy to the edge device
deploy_result = edge_deployer.deploy(
    registry_image=registry_image,
    edge_device=edge_device,
    manifest=manifest
)

if deploy_result.success:
    print(f"Coalition '{coalition.name}' deployed successfully to {edge_device['name']}")
    print(f"Deployment ID: {deploy_result.deployment_id}")
else:
    print(f"Failed to deploy coalition '{coalition.name}' to {edge_device['name']}")
    print(f"Error: {deploy_result.error}")
    # Exit if deployment failed
    import sys
    sys.exit(1)
```

## Step 10: Verify Deployment

Verify that the deployment is functioning correctly on the edge device.

```python
# Create a deployment verifier
verifier = DeploymentVerifier()

# Verify the deployment
verification_result = verifier.verify_deployment(
    deployment_id=deploy_result.deployment_id,
    edge_device=edge_device
)

if verification_result.success:
    print(f"Deployment verification successful")
    print(f"Coalition '{coalition.name}' is running on {edge_device['name']}")
    print(f"Health check status: {verification_result.health_status}")
    print(f"Resource usage: {verification_result.resource_usage}")
else:
    print(f"Deployment verification failed")
    print(f"Error: {verification_result.error}")
    # Exit if verification failed
    import sys
    sys.exit(1)
```

## Step 11: Monitor the Deployed Coalition

Set up monitoring for the deployed coalition to ensure it continues to function correctly.

```python
from infrastructure.monitoring.edge_monitor import EdgeMonitor

# Create an edge monitor
monitor = EdgeMonitor()

# Set up monitoring for the deployment
monitoring_result = monitor.setup_monitoring(
    deployment_id=deploy_result.deployment_id,
    edge_device=edge_device,
    metrics=["cpu", "memory", "network", "agent_status", "coalition_health"],
    alert_thresholds={
        "cpu_usage": 90,  # Alert if CPU usage exceeds 90%
        "memory_usage": 85,  # Alert if memory usage exceeds 85%
        "coalition_health": 70  # Alert if coalition health drops below 70%
    },
    alert_channels=["email", "webhook"]
)

if monitoring_result.success:
    print(f"Monitoring set up successfully")
    print(f"Dashboard URL: {monitoring_result.dashboard_url}")
else:
    print(f"Failed to set up monitoring")
    print(f"Error: {monitoring_result.error}")
```

## Complete Example

Here's a complete example that puts everything together:

```python
from coalitions.deployment.edge_packager import EdgePackager
from coalitions.deployment.deployment_manifest import DeploymentManifest
from infrastructure.export.coalition_packaging import CoalitionPackager
from infrastructure.deployment.hardware_compatibility import HardwareCompatibilityChecker
from infrastructure.deployment.deployment_verification import DeploymentVerifier
from infrastructure.deployment.edge_deployer import EdgeDeployer
from infrastructure.monitoring.edge_monitor import EdgeMonitor
from readiness.readiness_evaluator import ReadinessEvaluator
import subprocess
import sys

# Assume we already have a coalition object from previous tutorials

# Step 1: Check coalition readiness
readiness_evaluator = ReadinessEvaluator()
readiness_result = readiness_evaluator.evaluate_coalition(coalition)

if not readiness_result.is_ready:
    print(f"Coalition '{coalition.name}' is not ready for deployment")
    print(f"Readiness score: {readiness_result.score}/100")
    print(f"Issues to address: {readiness_result.issues}")
    sys.exit(1)

print(f"Coalition '{coalition.name}' is ready for deployment")
print(f"Readiness score: {readiness_result.score}/100")

# Step 2: Define target device and check compatibility
target_device = {
    "type": "raspberry_pi_4b",
    "cpu": "ARM Cortex-A72",
    "ram": 4,  # GB
    "storage": 32,  # GB
    "gpu": None,
    "network": ["wifi", "ethernet"],
    "sensors": ["camera", "microphone"]
}

compatibility_checker = HardwareCompatibilityChecker()
compatibility_result = compatibility_checker.check_compatibility(coalition, target_device)

if not compatibility_result.is_compatible:
    print(f"Coalition '{coalition.name}' is not compatible with the target device")
    print(f"Compatibility issues: {compatibility_result.issues}")
    sys.exit(1)

print(f"Coalition '{coalition.name}' is compatible with the target device")

# Step 3: Create deployment manifest
manifest = DeploymentManifest(
    coalition_id=coalition.id,
    coalition_name=coalition.name,
    version="1.0.0",
    description="Resource gathering coalition for edge deployment",
    required_capabilities={
        "cpu": "ARM Cortex-A72",
        "ram_min": 2,  # GB
        "storage_min": 16,  # GB
        "network": ["wifi"],
        "sensors": ["camera"]
    },
    environment_variables={
        "LOG_LEVEL": "INFO",
        "MAX_ENERGY_CONSUMPTION": "50",
        "DATA_COLLECTION_INTERVAL": "300"
    },
    persistence={
        "enabled": True,
        "backup_interval": 3600,  # seconds
        "max_storage": 1  # GB
    },
    security={
        "encryption": "AES-256",
        "secure_boot": True,
        "access_control": "role_based"
    }
)

# Step 4: Package the coalition
packager = CoalitionPackager()
package_result = packager.package_coalition(
    coalition=coalition,
    manifest=manifest,
    output_dir="./output",
    target_platform=target_device["type"]
)

if not package_result.success:
    print(f"Failed to package coalition '{coalition.name}'")
    print(f"Error: {package_result.error}")
    sys.exit(1)

print(f"Coalition '{coalition.name}' packaged successfully")
print(f"Package location: {package_result.package_path}")

# Step 5: Create Docker container
edge_packager = EdgePackager()
container_result = edge_packager.create_container(
    package_path=package_result.package_path,
    manifest=manifest,
    target_platform=target_device["type"],
    output_dir="./containers"
)

if not container_result.success:
    print(f"Failed to create Docker container")
    print(f"Error: {container_result.error}")
    sys.exit(1)

print(f"Docker container created successfully")
print(f"Container image: {container_result.image_name}")

# Step 6: Test the container locally
print("Testing container locally...")
try:
    result = subprocess.run(
        ["docker", "run", "--rm", "-p", "8080:8080", container_result.image_name],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode != 0:
        print(f"Container test failed with return code {result.returncode}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print("Container test successful")
except Exception as e:
    print(f"Error testing container: {str(e)}")
    sys.exit(1)

# Step 7: Push container to registry
print("Pushing container to registry...")
try:
    registry_url = "registry.example.com"
    registry_image = f"{registry_url}/{container_result.image_name}"

    subprocess.run(
        ["docker", "tag", container_result.image_name, registry_image],
        check=True
    )

    subprocess.run(
        ["docker", "push", registry_image],
        check=True
    )

    print(f"Container pushed to registry: {registry_image}")
except Exception as e:
    print(f"Error pushing container to registry: {str(e)}")
    sys.exit(1)

# Step 8: Deploy to edge device
edge_device = {
    "name": "raspberry-pi-01",
    "host": "192.168.1.100",
    "port": 22,
    "username": "pi",
    "key_file": "~/.ssh/id_rsa",
    "platform": "raspberry_pi_4b"
}

edge_deployer = EdgeDeployer()
deploy_result = edge_deployer.deploy(
    registry_image=registry_image,
    edge_device=edge_device,
    manifest=manifest
)

if not deploy_result.success:
    print(f"Failed to deploy coalition '{coalition.name}' to {edge_device['name']}")
    print(f"Error: {deploy_result.error}")
    sys.exit(1)

print(f"Coalition '{coalition.name}' deployed successfully to {edge_device['name']}")
print(f"Deployment ID: {deploy_result.deployment_id}")

# Step 9: Verify deployment
verifier = DeploymentVerifier()
verification_result = verifier.verify_deployment(
    deployment_id=deploy_result.deployment_id,
    edge_device=edge_device
)

if not verification_result.success:
    print(f"Deployment verification failed")
    print(f"Error: {verification_result.error}")
    sys.exit(1)

print(f"Deployment verification successful")
print(f"Coalition '{coalition.name}' is running on {edge_device['name']}")
print(f"Health check status: {verification_result.health_status}")

# Step 10: Set up monitoring
monitor = EdgeMonitor()
monitoring_result = monitor.setup_monitoring(
    deployment_id=deploy_result.deployment_id,
    edge_device=edge_device,
    metrics=["cpu", "memory", "network", "agent_status", "coalition_health"],
    alert_thresholds={
        "cpu_usage": 90,
        "memory_usage": 85,
        "coalition_health": 70
    },
    alert_channels=["email", "webhook"]
)

if monitoring_result.success:
    print(f"Monitoring set up successfully")
    print(f"Dashboard URL: {monitoring_result.dashboard_url}")
else:
    print(f"Failed to set up monitoring")
    print(f"Error: {monitoring_result.error}")

print(f"Coalition '{coalition.name}' has been successfully deployed to the edge device")
print("Deployment process completed")
```

## Next Steps

Now that you've deployed a coalition to an edge device, you might want to:

1. [Set up remote management](remote-management.md) to control and update the deployed coalition
2. [Implement data synchronization](data-synchronization.md) between edge devices and the central system
3. [Create a coalition mesh network](coalition-mesh-network.md) for communication between multiple deployed coalitions

## Troubleshooting

### Common Issues

1. **Deployment fails**: Check network connectivity, SSH credentials, and Docker installation on the edge device
2. **Container doesn't start**: Verify hardware compatibility and resource requirements
3. **Coalition doesn't function properly**: Check logs for errors and ensure all dependencies are included in the container
4. **Monitoring issues**: Verify that the monitoring agent is running and can connect to the monitoring server

### Getting Help

If you encounter issues not covered here, check the [API Reference](../api/index.md) or ask for help in the [FreeAgentics community forum](https://community.freeagentics.ai).
