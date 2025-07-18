"""Enhanced environment orchestrator with resource pool management and isolation."""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import docker
import yaml

from .environment_manager import EnvironmentManager
from .test_isolation import IsolationLevel, TestIsolation

logger = logging.getLogger(__name__)


class EnvironmentProfile(Enum):
    """Test environment profiles."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    LOAD = "load"
    SECURITY = "security"


@dataclass
class ResourcePool:
    """Resource pool configuration."""

    name: str
    min_instances: int
    max_instances: int
    current_instances: int = 0
    available_instances: Set[str] = None
    busy_instances: Set[str] = None

    def __post_init__(self):
        if self.available_instances is None:
            self.available_instances = set()
        if self.busy_instances is None:
            self.busy_instances = set()


@dataclass
class TestEnvironmentSpec:
    """Test environment specification."""

    name: str
    profile: EnvironmentProfile
    services: List[str]
    resources: Dict[str, Any]
    isolation_level: IsolationLevel
    parallel_instances: int = 1
    timeout: int = 300
    cleanup_on_exit: bool = True


class EnvironmentOrchestrator:
    """Enhanced environment orchestrator with resource pooling and isolation."""

    def __init__(
        self, config_path: str = "tests/environment/orchestrator_config.yml"
    ):
        self.config_path = config_path
        self.config = self._load_config()
        self.resource_pools = {}
        self.active_environments = {}
        self.isolation_manager = None
        self.docker_client = docker.from_env()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.Lock()

        # Initialize resource pools
        self._initialize_resource_pools()

        # Initialize isolation manager
        self._initialize_isolation_manager()

    def _load_config(self) -> Dict[str, Any]:
        """Load orchestrator configuration."""
        if Path(self.config_path).exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                "resource_pools": {
                    "postgres": {"min_instances": 2, "max_instances": 5},
                    "redis": {"min_instances": 1, "max_instances": 3},
                    "rabbitmq": {"min_instances": 1, "max_instances": 3},
                    "elasticsearch": {"min_instances": 1, "max_instances": 2},
                    "minio": {"min_instances": 1, "max_instances": 2},
                },
                "profiles": {
                    "unit": {
                        "services": ["postgres", "redis"],
                        "isolation_level": "SCHEMA",
                        "parallel_instances": 3,
                        "timeout": 60,
                    },
                    "integration": {
                        "services": ["postgres", "redis", "rabbitmq"],
                        "isolation_level": "SCHEMA",
                        "parallel_instances": 2,
                        "timeout": 120,
                    },
                    "e2e": {
                        "services": [
                            "postgres",
                            "redis",
                            "rabbitmq",
                            "elasticsearch",
                        ],
                        "isolation_level": "DATABASE",
                        "parallel_instances": 1,
                        "timeout": 300,
                    },
                    "performance": {
                        "services": [
                            "postgres",
                            "redis",
                            "rabbitmq",
                            "elasticsearch",
                            "minio",
                        ],
                        "isolation_level": "CONTAINER",
                        "parallel_instances": 1,
                        "timeout": 600,
                    },
                },
                "cleanup": {
                    "max_age_hours": 24,
                    "auto_cleanup_interval": 3600,
                },
            }

    def _initialize_resource_pools(self):
        """Initialize resource pools."""
        for pool_name, pool_config in self.config["resource_pools"].items():
            self.resource_pools[pool_name] = ResourcePool(
                name=pool_name,
                min_instances=pool_config["min_instances"],
                max_instances=pool_config["max_instances"],
            )

    def _initialize_isolation_manager(self):
        """Initialize test isolation manager."""
        isolation_config = {
            "postgres": {
                "host": "localhost",
                "port": 5433,
                "user": "test_user",
                "password": "test_password",
                "database": "freeagentics_test",
            },
            "redis": {"host": "localhost", "port": 6380},
            "rabbitmq": {
                "host": "localhost",
                "port": 5673,
                "user": "test_user",
                "password": "test_password",
            },
            "filesystem_base": "/tmp/test_isolation",
        }

        self.isolation_manager = TestIsolation(isolation_config)

    def create_environment(self, spec: TestEnvironmentSpec) -> str:
        """Create a new test environment based on specification."""
        with self._lock:
            env_id = f"{spec.name}_{int(time.time())}"

            try:
                # Allocate resources from pools
                allocated_resources = self._allocate_resources(spec)

                # Create environment manager
                env_manager = EnvironmentManager(
                    compose_file="docker-compose.test.yml",
                    project_name=f"test_{env_id}",
                    work_dir=os.getcwd(),
                )

                # Start environment
                if env_manager.start(services=spec.services):
                    # Set up isolation
                    isolation_context = self.isolation_manager.isolate_all(
                        env_id
                    )

                    # Store environment
                    self.active_environments[env_id] = {
                        "spec": spec,
                        "manager": env_manager,
                        "resources": allocated_resources,
                        "isolation_context": isolation_context,
                        "created_at": time.time(),
                        "status": "running",
                    }

                    logger.info(
                        f"Created environment {env_id} with profile {spec.profile.value}"
                    )
                    return env_id
                else:
                    # Release resources if environment failed to start
                    self._release_resources(allocated_resources)
                    raise Exception(f"Failed to start environment {env_id}")

            except Exception as e:
                logger.error(f"Error creating environment {env_id}: {e}")
                raise

    def _allocate_resources(self, spec: TestEnvironmentSpec) -> Dict[str, Any]:
        """Allocate resources from pools for the environment."""
        allocated = {}

        for service in spec.services:
            if service in self.resource_pools:
                pool = self.resource_pools[service]

                if len(pool.available_instances) > 0:
                    # Use available instance
                    instance_id = pool.available_instances.pop()
                    pool.busy_instances.add(instance_id)
                elif pool.current_instances < pool.max_instances:
                    # Create new instance
                    instance_id = f"{service}_{pool.current_instances + 1}"
                    pool.current_instances += 1
                    pool.busy_instances.add(instance_id)
                else:
                    # No resources available
                    self._release_resources(allocated)
                    raise Exception(
                        f"No resources available for service {service}"
                    )

                allocated[service] = instance_id

        return allocated

    def _release_resources(self, allocated_resources: Dict[str, Any]):
        """Release allocated resources back to pools."""
        for service, instance_id in allocated_resources.items():
            if service in self.resource_pools:
                pool = self.resource_pools[service]
                if instance_id in pool.busy_instances:
                    pool.busy_instances.remove(instance_id)
                    pool.available_instances.add(instance_id)

    def destroy_environment(self, env_id: str) -> bool:
        """Destroy a test environment and release resources."""
        with self._lock:
            if env_id not in self.active_environments:
                logger.warning(f"Environment {env_id} not found")
                return False

            env_info = self.active_environments[env_id]

            try:
                # Stop environment
                env_info["manager"].stop()

                # Clean up isolation
                self.isolation_manager.cleanup_all(
                    env_info["isolation_context"]
                )

                # Release resources
                self._release_resources(env_info["resources"])

                # Remove from active environments
                del self.active_environments[env_id]

                logger.info(f"Destroyed environment {env_id}")
                return True

            except Exception as e:
                logger.error(f"Error destroying environment {env_id}: {e}")
                return False

    @contextmanager
    def environment(self, spec: TestEnvironmentSpec):
        """Context manager for test environment."""
        env_id = self.create_environment(spec)
        try:
            yield env_id, self.active_environments[env_id]
        finally:
            if spec.cleanup_on_exit:
                self.destroy_environment(env_id)

    def get_environment_info(self, env_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an environment."""
        return self.active_environments.get(env_id)

    def list_environments(self) -> List[str]:
        """List all active environments."""
        return list(self.active_environments.keys())

    def get_pool_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all resource pools."""
        status = {}

        for pool_name, pool in self.resource_pools.items():
            status[pool_name] = {
                "current_instances": pool.current_instances,
                "available_instances": len(pool.available_instances),
                "busy_instances": len(pool.busy_instances),
                "max_instances": pool.max_instances,
                "utilization": (
                    len(pool.busy_instances) / pool.max_instances
                    if pool.max_instances > 0
                    else 0
                ),
            }

        return status

    def scale_pool(self, pool_name: str, target_instances: int) -> bool:
        """Scale a resource pool to target instances."""
        if pool_name not in self.resource_pools:
            return False

        pool = self.resource_pools[pool_name]

        with self._lock:
            if target_instances > pool.max_instances:
                target_instances = pool.max_instances

            if target_instances > pool.current_instances:
                # Scale up
                for i in range(pool.current_instances, target_instances):
                    instance_id = f"{pool_name}_{i + 1}"
                    pool.available_instances.add(instance_id)

                pool.current_instances = target_instances
                logger.info(
                    f"Scaled up pool {pool_name} to {target_instances} instances"
                )

            elif target_instances < pool.current_instances:
                # Scale down (only if instances are available)
                instances_to_remove = pool.current_instances - target_instances
                available_count = len(pool.available_instances)
                busy_count = len(pool.busy_instances)

                # Check if we can scale down without affecting busy instances
                # We need at least as many instances as we have busy ones
                if target_instances <= busy_count:
                    logger.warning(
                        f"Cannot scale down pool {pool_name}: would remove busy instances"
                    )
                    return False

                if instances_to_remove <= available_count:
                    for _ in range(instances_to_remove):
                        if pool.available_instances:
                            pool.available_instances.pop()

                    pool.current_instances = target_instances
                    logger.info(
                        f"Scaled down pool {pool_name} to {target_instances} instances"
                    )
                else:
                    logger.warning(
                        f"Cannot scale down pool {pool_name}: not enough available instances"
                    )
                    return False

        return True

    def run_parallel_tests(
        self, specs: List[TestEnvironmentSpec]
    ) -> Dict[str, Any]:
        """Run multiple test environments in parallel."""
        results = {}
        futures = []

        for spec in specs:
            future = self.executor.submit(self._run_single_environment, spec)
            futures.append((spec.name, future))

        for spec_name, future in futures:
            try:
                result = future.result(timeout=600)  # 10 minute timeout
                results[spec_name] = result
            except Exception as e:
                results[spec_name] = {"error": str(e)}

        return results

    def _run_single_environment(
        self, spec: TestEnvironmentSpec
    ) -> Dict[str, Any]:
        """Run a single test environment."""
        try:
            with self.environment(spec) as (env_id, env_info):
                # Environment is running, return info
                return {
                    "env_id": env_id,
                    "status": "success",
                    "started_at": env_info["created_at"],
                    "resources": env_info["resources"],
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cleanup_orphaned_resources(self) -> Dict[str, Any]:
        """Clean up orphaned Docker containers, volumes, and networks."""
        cleanup_results = {
            "containers": 0,
            "volumes": 0,
            "networks": 0,
            "images": 0,
        }

        try:
            # Clean up test containers
            containers = self.docker_client.containers.list(all=True)
            for container in containers:
                if "test" in container.name and container.status == "exited":
                    container.remove()
                    cleanup_results["containers"] += 1

            # Clean up test volumes
            volumes = self.docker_client.volumes.list()
            for volume in volumes:
                if "test" in volume.name:
                    try:
                        volume.remove()
                        cleanup_results["volumes"] += 1
                    except:
                        pass  # Volume might be in use

            # Clean up test networks
            networks = self.docker_client.networks.list()
            for network in networks:
                if "test" in network.name and network.name != "bridge":
                    try:
                        network.remove()
                        cleanup_results["networks"] += 1
                    except:
                        pass  # Network might be in use

            # Clean up dangling images
            images = self.docker_client.images.list(filters={"dangling": True})
            for image in images:
                try:
                    self.docker_client.images.remove(image.id)
                    cleanup_results["images"] += 1
                except:
                    pass

            # Clean up isolation resources
            self.isolation_manager.cleanup_orphaned_resources()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        return cleanup_results

    def auto_cleanup(self):
        """Automatically clean up old environments."""
        current_time = time.time()
        max_age = self.config["cleanup"]["max_age_hours"] * 3600

        environments_to_cleanup = []

        for env_id, env_info in self.active_environments.items():
            if current_time - env_info["created_at"] > max_age:
                environments_to_cleanup.append(env_id)

        for env_id in environments_to_cleanup:
            logger.info(f"Auto-cleaning up old environment {env_id}")
            self.destroy_environment(env_id)

        # Clean up orphaned resources
        self.cleanup_orphaned_resources()

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        return {
            "active_environments": len(self.active_environments),
            "resource_pools": self.get_pool_status(),
            "uptime": time.time() - getattr(self, "_start_time", time.time()),
            "total_environments_created": getattr(self, "_total_created", 0),
            "total_environments_destroyed": getattr(
                self, "_total_destroyed", 0
            ),
        }

    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def __enter__(self):
        """Context manager entry."""
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Clean up all active environments
        for env_id in list(self.active_environments.keys()):
            self.destroy_environment(env_id)

        # Clean up orphaned resources
        self.cleanup_orphaned_resources()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        return False


# Environment profile factory functions
def create_unit_test_spec(name: str, **kwargs) -> TestEnvironmentSpec:
    """Create a unit test environment specification."""
    defaults = {
        "profile": EnvironmentProfile.UNIT,
        "services": ["postgres", "redis"],
        "resources": {},
        "isolation_level": IsolationLevel.SCHEMA,
        "parallel_instances": 3,
        "timeout": 60,
    }
    defaults.update(kwargs)
    return TestEnvironmentSpec(name=name, **defaults)


def create_integration_test_spec(name: str, **kwargs) -> TestEnvironmentSpec:
    """Create an integration test environment specification."""
    defaults = {
        "profile": EnvironmentProfile.INTEGRATION,
        "services": ["postgres", "redis", "rabbitmq"],
        "resources": {},
        "isolation_level": IsolationLevel.SCHEMA,
        "parallel_instances": 2,
        "timeout": 120,
    }
    defaults.update(kwargs)
    return TestEnvironmentSpec(name=name, **defaults)


def create_e2e_test_spec(name: str, **kwargs) -> TestEnvironmentSpec:
    """Create an end-to-end test environment specification."""
    defaults = {
        "profile": EnvironmentProfile.E2E,
        "services": ["postgres", "redis", "rabbitmq", "elasticsearch"],
        "resources": {},
        "isolation_level": IsolationLevel.DATABASE,
        "parallel_instances": 1,
        "timeout": 300,
    }
    defaults.update(kwargs)
    return TestEnvironmentSpec(name=name, **defaults)


def create_performance_test_spec(name: str, **kwargs) -> TestEnvironmentSpec:
    """Create a performance test environment specification."""
    defaults = {
        "profile": EnvironmentProfile.PERFORMANCE,
        "services": [
            "postgres",
            "redis",
            "rabbitmq",
            "elasticsearch",
            "minio",
        ],
        "resources": {},
        "isolation_level": IsolationLevel.CONTAINER,
        "parallel_instances": 1,
        "timeout": 600,
    }
    defaults.update(kwargs)
    return TestEnvironmentSpec(name=name, **defaults)
