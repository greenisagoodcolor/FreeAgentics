"""Tests for the Environment Manager."""

from unittest.mock import Mock, patch

import psycopg2
import pytest

import redis
from tests.environment.environment_manager import (
    EnvironmentManager,
    EnvironmentState,
    PortAllocator,
    ServiceHealth,
)


class TestPortAllocator:
    """Test the port allocation functionality."""

    def test_allocate_port_returns_unique_ports(self):
        """Test that port allocator returns unique ports."""
        allocator = PortAllocator(base_port=50000)

        port1 = allocator.allocate_port("service1")
        port2 = allocator.allocate_port("service2")

        assert port1 != port2
        assert port1 >= 50000
        assert port2 >= 50000

    def test_allocate_port_for_same_service_returns_same_port(self):
        """Test that same service gets same port."""
        allocator = PortAllocator(base_port=50000)

        port1 = allocator.allocate_port("service1")
        port2 = allocator.allocate_port("service1")

        assert port1 == port2

    def test_deallocate_port_frees_port(self):
        """Test that deallocating port makes it available again."""
        allocator = PortAllocator(base_port=50000)

        port1 = allocator.allocate_port("service1")
        allocator.deallocate_port("service1")

        # Should be able to allocate the same port for a different service
        port2 = allocator.allocate_port("service2")
        assert port2 == port1

    def test_get_allocated_ports(self):
        """Test getting all allocated ports."""
        allocator = PortAllocator(base_port=50000)

        allocator.allocate_port("service1")
        allocator.allocate_port("service2")

        allocated = allocator.get_allocated_ports()
        assert len(allocated) == 2
        assert "service1" in allocated
        assert "service2" in allocated


class TestServiceHealth:
    """Test service health checking."""

    def test_check_postgres_health_success(self):
        """Test PostgreSQL health check when healthy."""
        health_checker = ServiceHealth()

        with patch("psycopg2.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            is_healthy = health_checker.check_postgres(
                host="localhost",
                port=5432,
                user="test",
                password="test",
                database="test",
            )

            assert is_healthy is True
            mock_conn.close.assert_called_once()

    def test_check_postgres_health_failure(self):
        """Test PostgreSQL health check when unhealthy."""
        health_checker = ServiceHealth()

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.side_effect = psycopg2.OperationalError(
                "Connection failed"
            )

            is_healthy = health_checker.check_postgres(
                host="localhost",
                port=5432,
                user="test",
                password="test",
                database="test",
            )

            assert is_healthy is False

    def test_check_redis_health_success(self):
        """Test Redis health check when healthy."""
        health_checker = ServiceHealth()

        with patch("redis.Redis") as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis

            is_healthy = health_checker.check_redis(
                host="localhost", port=6379
            )

            assert is_healthy is True
            mock_redis.close.assert_called_once()

    def test_check_redis_health_failure(self):
        """Test Redis health check when unhealthy."""
        health_checker = ServiceHealth()

        with patch("redis.Redis") as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.side_effect = redis.ConnectionError(
                "Connection failed"
            )
            mock_redis_class.return_value = mock_redis

            is_healthy = health_checker.check_redis(
                host="localhost", port=6379
            )

            assert is_healthy is False


class TestEnvironmentManager:
    """Test the main environment manager."""

    @pytest.fixture
    def env_manager(self, tmp_path):
        """Create an environment manager instance."""
        return EnvironmentManager(
            compose_file="docker-compose.test.yml",
            project_name="test_project",
            work_dir=str(tmp_path),
        )

    def test_initialization(self, env_manager):
        """Test environment manager initialization."""
        assert env_manager.state == EnvironmentState.STOPPED
        assert env_manager.project_name == "test_project"
        assert env_manager.compose_file == "docker-compose.test.yml"

    @patch("docker.from_env")
    def test_start_environment_success(self, mock_docker, env_manager):
        """Test starting the environment successfully."""
        # Mock docker client
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # Mock compose operations
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            # Mock health checks
            with patch.object(env_manager, "_wait_for_services") as mock_wait:
                mock_wait.return_value = True

                result = env_manager.start()

                assert result is True
                assert env_manager.state == EnvironmentState.RUNNING

                # Verify docker-compose was called
                mock_run.assert_called()
                call_args = mock_run.call_args[0][0]
                assert "docker-compose" in call_args
                assert "up" in call_args
                assert "-d" in call_args

    @patch("docker.from_env")
    def test_start_environment_health_check_failure(
        self, mock_docker, env_manager
    ):
        """Test starting environment when health checks fail."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            with patch.object(env_manager, "_wait_for_services") as mock_wait:
                mock_wait.return_value = False

                result = env_manager.start()

                assert result is False
                assert env_manager.state == EnvironmentState.ERROR

    def test_stop_environment(self, env_manager):
        """Test stopping the environment."""
        # Set initial state to running
        env_manager.state = EnvironmentState.RUNNING

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = env_manager.stop()

            assert result is True
            assert env_manager.state == EnvironmentState.STOPPED

            # Verify docker-compose was called
            mock_run.assert_called()
            call_args = mock_run.call_args[0][0]
            assert "docker-compose" in call_args
            assert "down" in call_args

    def test_reset_environment(self, env_manager):
        """Test resetting the environment."""
        env_manager.state = EnvironmentState.RUNNING

        with patch.object(env_manager, "stop") as mock_stop:
            mock_stop.return_value = True

            with patch.object(env_manager, "_clean_volumes") as mock_clean:
                mock_clean.return_value = True

                with patch.object(env_manager, "start") as mock_start:
                    mock_start.return_value = True

                    result = env_manager.reset()

                    assert result is True
                    mock_stop.assert_called_once()
                    mock_clean.assert_called_once()
                    mock_start.assert_called_once()

    def test_get_service_config(self, env_manager):
        """Test getting service configuration."""
        # Mock port allocator
        with patch.object(
            env_manager.port_allocator, "allocate_port"
        ) as mock_allocate:
            mock_allocate.side_effect = [
                5433,  # postgres port
                6380,  # redis port
                6380,  # redis url
                5673,  # rabbitmq port
                9201,  # elasticsearch port
                9201,  # elasticsearch url
                9002,  # minio port
                9002,  # minio endpoint_url
            ]  # services with URLs call allocate_port twice

            config = env_manager.get_service_config()

            assert "postgres" in config
            assert config["postgres"]["port"] == 5433
            assert config["redis"]["port"] == 6380
            assert config["rabbitmq"]["port"] == 5673
            assert config["elasticsearch"]["port"] == 9201
            assert config["minio"]["port"] == 9002

    def test_validate_state(self, env_manager):
        """Test environment state validation."""
        env_manager.state = EnvironmentState.RUNNING

        with patch.object(
            env_manager.health_checker, "check_postgres"
        ) as mock_pg:
            mock_pg.return_value = True

            with patch.object(
                env_manager.health_checker, "check_redis"
            ) as mock_redis:
                mock_redis.return_value = True

                with patch.object(
                    env_manager.health_checker, "check_all_services"
                ) as mock_all:
                    mock_all.return_value = {"postgres": True, "redis": True}

                    is_valid = env_manager.validate_state()

                    assert is_valid is True

    def test_context_manager(self, env_manager):
        """Test using environment manager as context manager."""
        with patch.object(env_manager, "start") as mock_start:
            mock_start.return_value = True

            with patch.object(env_manager, "stop") as mock_stop:
                mock_stop.return_value = True

                with env_manager as em:
                    assert em == env_manager
                    mock_start.assert_called_once()

                mock_stop.assert_called_once()

    def test_context_manager_handles_exceptions(self, env_manager):
        """Test context manager handles exceptions properly."""
        with patch.object(env_manager, "start") as mock_start:
            mock_start.return_value = True

            with patch.object(env_manager, "stop") as mock_stop:
                mock_stop.return_value = True

                try:
                    with env_manager:
                        raise ValueError("Test exception")
                except ValueError:
                    pass

                # Stop should still be called
                mock_stop.assert_called_once()

    def test_get_logs(self, env_manager):
        """Test getting service logs."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0, stdout="Service logs here", stderr=""
            )

            logs = env_manager.get_logs("postgres", lines=100)

            assert logs == "Service logs here"
            mock_run.assert_called()
            call_args = mock_run.call_args[0][0]
            assert "docker-compose" in call_args
            assert "logs" in call_args
            assert "--tail=100" in call_args
            assert "postgres" in call_args
