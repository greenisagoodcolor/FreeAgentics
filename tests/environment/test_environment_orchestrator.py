"""Tests for the environment orchestrator."""

import time
from unittest.mock import Mock, patch

import pytest

from tests.environment.environment_orchestrator import (
    EnvironmentOrchestrator,
    EnvironmentProfile,
    ResourcePool,
    TestEnvironmentSpec,
    create_e2e_test_spec,
    create_integration_test_spec,
    create_performance_test_spec,
    create_unit_test_spec,
)
from tests.environment.test_isolation import IsolationLevel


class TestResourcePool:
    """Test resource pool functionality."""

    def test_resource_pool_initialization(self):
        """Test resource pool initialization."""
        pool = ResourcePool(name="test_pool", min_instances=2, max_instances=5)

        assert pool.name == "test_pool"
        assert pool.min_instances == 2
        assert pool.max_instances == 5
        assert pool.current_instances == 0
        assert len(pool.available_instances) == 0
        assert len(pool.busy_instances) == 0

    def test_resource_pool_sets_creation(self):
        """Test resource pool creates sets if not provided."""
        pool = ResourcePool(
            name="test_pool",
            min_instances=1,
            max_instances=3,
            available_instances=None,
            busy_instances=None,
        )

        assert isinstance(pool.available_instances, set)
        assert isinstance(pool.busy_instances, set)


class TestTestEnvironmentSpec:
    """Test environment specification."""

    def test_spec_creation(self):
        """Test creating a test environment specification."""
        spec = TestEnvironmentSpec(
            name="test_env",
            profile=EnvironmentProfile.UNIT,
            services=["postgres", "redis"],
            resources={"memory": "1g"},
            isolation_level=IsolationLevel.SCHEMA,
        )

        assert spec.name == "test_env"
        assert spec.profile == EnvironmentProfile.UNIT
        assert spec.services == ["postgres", "redis"]
        assert spec.resources == {"memory": "1g"}
        assert spec.isolation_level == IsolationLevel.SCHEMA
        assert spec.parallel_instances == 1
        assert spec.timeout == 300
        assert spec.cleanup_on_exit == True


class TestEnvironmentOrchestrator:
    """Test environment orchestrator functionality."""

    @pytest.fixture
    def mock_config_path(self, tmp_path):
        """Create a mock config path."""
        return str(tmp_path / "test_config.yml")

    @pytest.fixture
    def orchestrator(self, mock_config_path):
        """Create an orchestrator instance."""
        with patch("docker.from_env") as mock_docker:
            mock_docker.return_value = Mock()

            with patch(
                "tests.environment.environment_orchestrator.TestIsolation"
            ) as mock_isolation:
                mock_isolation.return_value = Mock()

                return EnvironmentOrchestrator(config_path=mock_config_path)

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.config is not None
        assert orchestrator.resource_pools is not None
        assert orchestrator.active_environments == {}
        assert orchestrator.isolation_manager is not None

    def test_load_default_config(self, orchestrator):
        """Test loading default configuration."""
        config = orchestrator.config

        assert "resource_pools" in config
        assert "profiles" in config
        assert "cleanup" in config

        # Check profiles
        assert "unit" in config["profiles"]
        assert "integration" in config["profiles"]
        assert "e2e" in config["profiles"]
        assert "performance" in config["profiles"]

    def test_initialize_resource_pools(self, orchestrator):
        """Test resource pool initialization."""
        pools = orchestrator.resource_pools

        assert "postgres" in pools
        assert "redis" in pools
        assert "rabbitmq" in pools

        postgres_pool = pools["postgres"]
        assert postgres_pool.name == "postgres"
        assert postgres_pool.min_instances == 2
        assert postgres_pool.max_instances == 5

    def test_allocate_resources_success(self, orchestrator):
        """Test successful resource allocation."""
        spec = TestEnvironmentSpec(
            name="test",
            profile=EnvironmentProfile.UNIT,
            services=["postgres", "redis"],
            resources={},
            isolation_level=IsolationLevel.SCHEMA,
        )

        # Pre-populate available instances
        orchestrator.resource_pools["postgres"].available_instances.add(
            "postgres_1"
        )
        orchestrator.resource_pools["redis"].available_instances.add("redis_1")

        allocated = orchestrator._allocate_resources(spec)

        assert "postgres" in allocated
        assert "redis" in allocated
        assert allocated["postgres"] == "postgres_1"
        assert allocated["redis"] == "redis_1"

        # Check instances moved to busy
        assert (
            "postgres_1"
            in orchestrator.resource_pools["postgres"].busy_instances
        )
        assert "redis_1" in orchestrator.resource_pools["redis"].busy_instances

    def test_allocate_resources_create_new_instance(self, orchestrator):
        """Test resource allocation creates new instances when needed."""
        spec = TestEnvironmentSpec(
            name="test",
            profile=EnvironmentProfile.UNIT,
            services=["postgres"],
            resources={},
            isolation_level=IsolationLevel.SCHEMA,
        )

        # No available instances, should create new one
        allocated = orchestrator._allocate_resources(spec)

        assert "postgres" in allocated
        assert allocated["postgres"] == "postgres_1"
        assert orchestrator.resource_pools["postgres"].current_instances == 1
        assert (
            "postgres_1"
            in orchestrator.resource_pools["postgres"].busy_instances
        )

    def test_allocate_resources_no_capacity(self, orchestrator):
        """Test resource allocation failure when no capacity."""
        spec = TestEnvironmentSpec(
            name="test",
            profile=EnvironmentProfile.UNIT,
            services=["postgres"],
            resources={},
            isolation_level=IsolationLevel.SCHEMA,
        )

        # Fill up the pool
        pool = orchestrator.resource_pools["postgres"]
        pool.current_instances = pool.max_instances
        for i in range(pool.max_instances):
            pool.busy_instances.add(f"postgres_{i+1}")

        with pytest.raises(Exception, match="No resources available"):
            orchestrator._allocate_resources(spec)

    def test_release_resources(self, orchestrator):
        """Test releasing resources back to pool."""
        allocated = {"postgres": "postgres_1", "redis": "redis_1"}

        # Add instances to busy sets
        orchestrator.resource_pools["postgres"].busy_instances.add(
            "postgres_1"
        )
        orchestrator.resource_pools["redis"].busy_instances.add("redis_1")

        orchestrator._release_resources(allocated)

        # Check instances moved to available
        assert (
            "postgres_1"
            in orchestrator.resource_pools["postgres"].available_instances
        )
        assert (
            "redis_1"
            in orchestrator.resource_pools["redis"].available_instances
        )
        assert (
            "postgres_1"
            not in orchestrator.resource_pools["postgres"].busy_instances
        )
        assert (
            "redis_1"
            not in orchestrator.resource_pools["redis"].busy_instances
        )

    def test_create_environment_success(self, orchestrator):
        """Test successful environment creation."""
        spec = TestEnvironmentSpec(
            name="test_env",
            profile=EnvironmentProfile.UNIT,
            services=["postgres", "redis"],
            resources={},
            isolation_level=IsolationLevel.SCHEMA,
        )

        # Mock environment manager
        mock_env_manager = Mock()
        mock_env_manager.start.return_value = True

        with patch(
            "tests.environment.environment_orchestrator.EnvironmentManager"
        ) as mock_env_class:
            mock_env_class.return_value = mock_env_manager

            # Mock isolation
            orchestrator.isolation_manager.isolate_all.return_value = {
                "test": "context"
            }

            # Pre-populate available instances
            orchestrator.resource_pools["postgres"].available_instances.add(
                "postgres_1"
            )
            orchestrator.resource_pools["redis"].available_instances.add(
                "redis_1"
            )

            env_id = orchestrator.create_environment(spec)

            assert env_id.startswith("test_env_")
            assert env_id in orchestrator.active_environments

            env_info = orchestrator.active_environments[env_id]
            assert env_info["spec"] == spec
            assert env_info["manager"] == mock_env_manager
            assert env_info["status"] == "running"

    def test_create_environment_failure(self, orchestrator):
        """Test environment creation failure."""
        spec = TestEnvironmentSpec(
            name="test_env",
            profile=EnvironmentProfile.UNIT,
            services=["postgres"],
            resources={},
            isolation_level=IsolationLevel.SCHEMA,
        )

        # Mock environment manager to fail
        mock_env_manager = Mock()
        mock_env_manager.start.return_value = False

        with patch(
            "tests.environment.environment_orchestrator.EnvironmentManager"
        ) as mock_env_class:
            mock_env_class.return_value = mock_env_manager

            # Pre-populate available instances
            orchestrator.resource_pools["postgres"].available_instances.add(
                "postgres_1"
            )

            with pytest.raises(Exception, match="Failed to start environment"):
                orchestrator.create_environment(spec)

            # Resources should be released
            assert (
                "postgres_1"
                in orchestrator.resource_pools["postgres"].available_instances
            )

    def test_destroy_environment_success(self, orchestrator):
        """Test successful environment destruction."""
        # Create mock environment
        env_id = "test_env_123"
        mock_env_manager = Mock()
        mock_env_manager.stop.return_value = True

        orchestrator.active_environments[env_id] = {
            "spec": Mock(),
            "manager": mock_env_manager,
            "resources": {"postgres": "postgres_1"},
            "isolation_context": {"test": "context"},
            "created_at": time.time(),
            "status": "running",
        }

        # Add resource to busy set
        orchestrator.resource_pools["postgres"].busy_instances.add(
            "postgres_1"
        )

        result = orchestrator.destroy_environment(env_id)

        assert result is True
        assert env_id not in orchestrator.active_environments
        assert (
            "postgres_1"
            in orchestrator.resource_pools["postgres"].available_instances
        )

        # Check isolation cleanup was called
        orchestrator.isolation_manager.cleanup_all.assert_called_once()

    def test_destroy_environment_not_found(self, orchestrator):
        """Test destroying non-existent environment."""
        result = orchestrator.destroy_environment("non_existent")

        assert result is False

    def test_environment_context_manager(self, orchestrator):
        """Test environment context manager."""
        spec = TestEnvironmentSpec(
            name="test_env",
            profile=EnvironmentProfile.UNIT,
            services=["postgres"],
            resources={},
            isolation_level=IsolationLevel.SCHEMA,
            cleanup_on_exit=True,
        )

        with patch.object(orchestrator, "create_environment") as mock_create:
            mock_create.return_value = "test_env_123"

            with patch.object(
                orchestrator, "destroy_environment"
            ) as mock_destroy:
                mock_destroy.return_value = True

                orchestrator.active_environments["test_env_123"] = {
                    "test": "info"
                }

                with orchestrator.environment(spec) as (env_id, env_info):
                    assert env_id == "test_env_123"
                    assert env_info == {"test": "info"}

                mock_destroy.assert_called_once_with("test_env_123")

    def test_get_pool_status(self, orchestrator):
        """Test getting pool status."""
        # Set up pool state
        pool = orchestrator.resource_pools["postgres"]
        pool.current_instances = 3
        pool.available_instances = {"postgres_1", "postgres_2"}
        pool.busy_instances = {"postgres_3"}

        status = orchestrator.get_pool_status()

        assert "postgres" in status
        postgres_status = status["postgres"]
        assert postgres_status["current_instances"] == 3
        assert postgres_status["available_instances"] == 2
        assert postgres_status["busy_instances"] == 1
        assert postgres_status["max_instances"] == 5
        assert postgres_status["utilization"] == 0.2  # 1/5

    def test_scale_pool_up(self, orchestrator):
        """Test scaling pool up."""
        result = orchestrator.scale_pool("postgres", 3)

        assert result is True

        pool = orchestrator.resource_pools["postgres"]
        assert pool.current_instances == 3
        assert len(pool.available_instances) == 3

    def test_scale_pool_down(self, orchestrator):
        """Test scaling pool down."""
        # First scale up
        orchestrator.scale_pool("postgres", 3)

        # Then scale down
        result = orchestrator.scale_pool("postgres", 1)

        assert result is True

        pool = orchestrator.resource_pools["postgres"]
        assert pool.current_instances == 1
        assert len(pool.available_instances) == 1

    def test_scale_pool_down_with_busy_instances(self, orchestrator):
        """Test scaling pool down when instances are busy."""
        # Scale up and make some instances busy
        orchestrator.scale_pool("postgres", 3)
        pool = orchestrator.resource_pools["postgres"]

        # Move instances to busy
        instance = pool.available_instances.pop()
        pool.busy_instances.add(instance)

        # Try to scale down too much
        result = orchestrator.scale_pool("postgres", 1)

        assert result is False  # Should fail

    def test_cleanup_orphaned_resources(self, orchestrator):
        """Test cleaning up orphaned resources."""
        # Mock Docker client
        mock_container = Mock()
        mock_container.name = "test_container"
        mock_container.status = "exited"
        mock_container.remove = Mock()

        orchestrator.docker_client.containers.list.return_value = [
            mock_container
        ]
        orchestrator.docker_client.volumes.list.return_value = []
        orchestrator.docker_client.networks.list.return_value = []
        orchestrator.docker_client.images.list.return_value = []

        result = orchestrator.cleanup_orphaned_resources()

        assert result["containers"] == 1
        mock_container.remove.assert_called_once()

    def test_auto_cleanup(self, orchestrator):
        """Test automatic cleanup of old environments."""
        # Create old environment
        env_id = "old_env_123"
        old_time = time.time() - (25 * 3600)  # 25 hours ago

        orchestrator.active_environments[env_id] = {
            "spec": Mock(),
            "manager": Mock(),
            "resources": {},
            "isolation_context": {},
            "created_at": old_time,
            "status": "running",
        }

        with patch.object(orchestrator, "destroy_environment") as mock_destroy:
            mock_destroy.return_value = True

            with patch.object(
                orchestrator, "cleanup_orphaned_resources"
            ) as mock_cleanup:
                mock_cleanup.return_value = {}

                orchestrator.auto_cleanup()

                mock_destroy.assert_called_once_with(env_id)
                mock_cleanup.assert_called_once()

    def test_get_metrics(self, orchestrator):
        """Test getting orchestrator metrics."""
        # Add some test data
        orchestrator.active_environments["test_env"] = {"test": "data"}
        orchestrator._start_time = time.time() - 3600  # 1 hour ago
        orchestrator._total_created = 10
        orchestrator._total_destroyed = 8

        metrics = orchestrator.get_metrics()

        assert metrics["active_environments"] == 1
        assert "resource_pools" in metrics
        assert metrics["uptime"] > 0
        assert metrics["total_environments_created"] == 10
        assert metrics["total_environments_destroyed"] == 8


class TestEnvironmentSpecFactories:
    """Test environment specification factory functions."""

    def test_create_unit_test_spec(self):
        """Test creating unit test specification."""
        spec = create_unit_test_spec("unit_test")

        assert spec.name == "unit_test"
        assert spec.profile == EnvironmentProfile.UNIT
        assert spec.services == ["postgres", "redis"]
        assert spec.isolation_level == IsolationLevel.SCHEMA
        assert spec.parallel_instances == 3
        assert spec.timeout == 60

    def test_create_integration_test_spec(self):
        """Test creating integration test specification."""
        spec = create_integration_test_spec("integration_test")

        assert spec.name == "integration_test"
        assert spec.profile == EnvironmentProfile.INTEGRATION
        assert spec.services == ["postgres", "redis", "rabbitmq"]
        assert spec.isolation_level == IsolationLevel.SCHEMA
        assert spec.parallel_instances == 2
        assert spec.timeout == 120

    def test_create_e2e_test_spec(self):
        """Test creating end-to-end test specification."""
        spec = create_e2e_test_spec("e2e_test")

        assert spec.name == "e2e_test"
        assert spec.profile == EnvironmentProfile.E2E
        assert spec.services == [
            "postgres",
            "redis",
            "rabbitmq",
            "elasticsearch",
        ]
        assert spec.isolation_level == IsolationLevel.DATABASE
        assert spec.parallel_instances == 1
        assert spec.timeout == 300

    def test_create_performance_test_spec(self):
        """Test creating performance test specification."""
        spec = create_performance_test_spec("performance_test")

        assert spec.name == "performance_test"
        assert spec.profile == EnvironmentProfile.PERFORMANCE
        assert spec.services == [
            "postgres",
            "redis",
            "rabbitmq",
            "elasticsearch",
            "minio",
        ]
        assert spec.isolation_level == IsolationLevel.CONTAINER
        assert spec.parallel_instances == 1
        assert spec.timeout == 600

    def test_spec_factory_with_overrides(self):
        """Test specification factory with parameter overrides."""
        spec = create_unit_test_spec(
            "custom_unit_test",
            timeout=120,
            parallel_instances=5,
            cleanup_on_exit=False,
        )

        assert spec.name == "custom_unit_test"
        assert spec.timeout == 120
        assert spec.parallel_instances == 5
        assert spec.cleanup_on_exit is False
        # Other defaults should still apply
        assert spec.profile == EnvironmentProfile.UNIT
        assert spec.services == ["postgres", "redis"]
