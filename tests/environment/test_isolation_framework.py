"""Tests for test isolation framework."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.environment.test_isolation import (
    DatabaseIsolation,
    FilesystemIsolation,
    IsolationLevel,
    IsolationTester,
    MessageQueueIsolation,
    RedisIsolation,
)


class TestDatabaseIsolation:
    """Test database isolation functionality."""

    @pytest.fixture
    def db_isolation(self):
        """Create database isolation instance."""
        return DatabaseIsolation(
            host="localhost",
            port=5432,
            user="test_user",
            password="test_password",
            database="test_db",
        )

    def test_create_isolated_schema(self, db_isolation):
        """Test creating an isolated schema."""
        with patch("psycopg2.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            schema_name = db_isolation.create_isolated_schema("test_suite")

            assert schema_name.startswith("test_suite_")
            mock_cursor.execute.assert_any_call(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
            mock_cursor.execute.assert_any_call(f"SET search_path TO {schema_name}")
            mock_conn.commit.assert_called()

    def test_create_isolated_database(self, db_isolation):
        """Test creating an isolated database."""
        with patch("psycopg2.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_conn.autocommit = True
            mock_connect.return_value = mock_conn

            db_name = db_isolation.create_isolated_database("test_suite")

            assert db_name.startswith("test_suite_")
            mock_cursor.execute.assert_any_call(f"CREATE DATABASE {db_name}")

    def test_cleanup_schema(self, db_isolation):
        """Test cleaning up a schema."""
        with patch("psycopg2.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            db_isolation.cleanup_schema("test_schema")

            mock_cursor.execute.assert_called_with("DROP SCHEMA IF EXISTS test_schema CASCADE")
            mock_conn.commit.assert_called()

    def test_cleanup_database(self, db_isolation):
        """Test cleaning up a database."""
        with patch("psycopg2.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_conn.autocommit = True
            mock_connect.return_value = mock_conn

            db_isolation.cleanup_database("test_db")

            # Should terminate connections first
            mock_cursor.execute.assert_any_call(
                """
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = %s
                AND pid <> pg_backend_pid()
            """,
                ("test_db",),
            )

            # Then drop database
            mock_cursor.execute.assert_any_call("DROP DATABASE IF EXISTS test_db")

    def test_context_manager_schema(self, db_isolation):
        """Test using schema isolation as context manager."""
        with patch.object(db_isolation, "create_isolated_schema") as mock_create:
            mock_create.return_value = "test_schema_123"

            with patch.object(db_isolation, "cleanup_schema") as mock_cleanup:
                with db_isolation.isolated_schema("test") as schema:
                    assert schema == "test_schema_123"

                mock_cleanup.assert_called_with("test_schema_123")

    def test_get_connection_url(self, db_isolation):
        """Test getting connection URL."""
        url = db_isolation.get_connection_url(schema="test_schema")

        assert "postgresql://" in url
        assert "test_user:test_password" in url
        assert "localhost:5432" in url
        assert "test_db" in url
        assert "options=--search_path%3Dtest_schema" in url


class TestRedisIsolation:
    """Test Redis isolation functionality."""

    @pytest.fixture
    def redis_isolation(self):
        """Create Redis isolation instance."""
        return RedisIsolation(host="localhost", port=6379)

    def test_create_isolated_namespace(self, redis_isolation):
        """Test creating an isolated namespace."""
        namespace = redis_isolation.create_isolated_namespace("test_suite")

        assert namespace.startswith("test:test_suite:")
        assert len(namespace) > len("test:test_suite:")

    def test_get_namespaced_client(self, redis_isolation):
        """Test getting a namespaced Redis client."""
        with patch("redis.Redis") as mock_redis_class:
            mock_redis = Mock()
            mock_redis_class.return_value = mock_redis

            namespace = "test:suite:123"
            client = redis_isolation.get_namespaced_client(namespace)

            # Should return a wrapped client
            assert hasattr(client, "namespace")
            assert client.namespace == namespace

    def test_cleanup_namespace(self, redis_isolation):
        """Test cleaning up a namespace."""
        with patch("redis.Redis") as mock_redis_class:
            mock_redis = Mock()
            mock_keys = [b"test:suite:123:key1", b"test:suite:123:key2"]
            mock_redis.scan_iter.return_value = mock_keys
            mock_redis_class.return_value = mock_redis

            redis_isolation.cleanup_namespace("test:suite:123")

            mock_redis.scan_iter.assert_called_with("test:suite:123:*")
            mock_redis.delete.assert_called_with(*mock_keys)

    def test_context_manager(self, redis_isolation):
        """Test using Redis isolation as context manager."""
        with patch.object(redis_isolation, "create_isolated_namespace") as mock_create:
            mock_create.return_value = "test:namespace:123"

            with patch.object(redis_isolation, "cleanup_namespace") as mock_cleanup:
                with redis_isolation.isolated_namespace("test") as namespace:
                    assert namespace == "test:namespace:123"

                mock_cleanup.assert_called_with("test:namespace:123")


class TestMessageQueueIsolation:
    """Test message queue isolation functionality."""

    @pytest.fixture
    def mq_isolation(self):
        """Create message queue isolation instance."""
        return MessageQueueIsolation(
            host="localhost",
            port=5672,
            user="test_user",
            password="test_password",
        )

    def test_create_virtual_host(self, mq_isolation):
        """Test creating a virtual host."""
        with patch("pika.BlockingConnection") as mock_conn_class:
            mock_conn = Mock()
            mock_channel = Mock()
            mock_conn.channel.return_value = mock_channel
            mock_conn_class.return_value = mock_conn

            with patch("requests.put") as mock_put:
                mock_put.return_value = Mock(status_code=201)

                vhost = mq_isolation.create_virtual_host("test_suite")

                assert vhost.startswith("test_suite_")
                mock_put.assert_called()

    def test_cleanup_virtual_host(self, mq_isolation):
        """Test cleaning up a virtual host."""
        with patch("requests.delete") as mock_delete:
            mock_delete.return_value = Mock(status_code=204)

            mq_isolation.cleanup_virtual_host("test_vhost")

            mock_delete.assert_called()

    def test_get_connection_params(self, mq_isolation):
        """Test getting connection parameters."""
        params = mq_isolation.get_connection_params("test_vhost")

        assert params.host == "localhost"
        assert params.port == 5672
        assert params.virtual_host == "test_vhost"

    def test_context_manager(self, mq_isolation):
        """Test using MQ isolation as context manager."""
        with patch.object(mq_isolation, "create_virtual_host") as mock_create:
            mock_create.return_value = "test_vhost_123"

            with patch.object(mq_isolation, "cleanup_virtual_host") as mock_cleanup:
                with mq_isolation.isolated_vhost("test") as vhost:
                    assert vhost == "test_vhost_123"

                mock_cleanup.assert_called_with("test_vhost_123")


class TestFilesystemIsolation:
    """Test filesystem isolation functionality."""

    @pytest.fixture
    def fs_isolation(self):
        """Create filesystem isolation instance."""
        return FilesystemIsolation(base_dir="/tmp/test_isolation")

    def test_create_sandbox(self, fs_isolation):
        """Test creating a filesystem sandbox."""
        with patch("tempfile.mkdtemp") as mock_mkdtemp:
            mock_mkdtemp.return_value = "/tmp/test_sandbox_123"

            sandbox = fs_isolation.create_sandbox("test_suite")

            assert sandbox == Path("/tmp/test_sandbox_123")
            mock_mkdtemp.assert_called_with(prefix="test_suite_", dir="/tmp/test_isolation")

    def test_cleanup_sandbox(self, fs_isolation):
        """Test cleaning up a sandbox."""
        with patch("shutil.rmtree") as mock_rmtree:
            with patch("pathlib.Path.exists", return_value=True):
                fs_isolation.cleanup_sandbox("/tmp/test_sandbox")

                mock_rmtree.assert_called_with(Path("/tmp/test_sandbox"), ignore_errors=True)

    def test_context_manager(self, fs_isolation):
        """Test using filesystem isolation as context manager."""
        with patch.object(fs_isolation, "create_sandbox") as mock_create:
            mock_create.return_value = Path("/tmp/sandbox_123")

            with patch.object(fs_isolation, "cleanup_sandbox") as mock_cleanup:
                with fs_isolation.sandboxed("test") as sandbox:
                    assert sandbox == Path("/tmp/sandbox_123")

                mock_cleanup.assert_called_with(Path("/tmp/sandbox_123"))

    def test_copy_to_sandbox(self, fs_isolation):
        """Test copying files to sandbox."""
        with patch("shutil.copytree"):
            with patch("shutil.copy2") as mock_copy2:
                with patch("os.path.isdir") as mock_isdir:
                    mock_isdir.return_value = False

                    fs_isolation.copy_to_sandbox(
                        "/source/file.txt", Path("/tmp/sandbox"), "dest.txt"
                    )

                    mock_copy2.assert_called_with(
                        Path("/source/file.txt"), Path("/tmp/sandbox/dest.txt")
                    )


class TestTestIsolation:
    """Test the main test isolation class."""

    @pytest.fixture
    def test_isolation(self):
        """Create test isolation instance."""
        config = {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "user": "test",
                "password": "test",
                "database": "test",
            },
            "redis": {"host": "localhost", "port": 6379},
            "rabbitmq": {
                "host": "localhost",
                "port": 5672,
                "user": "test",
                "password": "test",
            },
        }
        return IsolationTester(config)

    def test_isolate_all(self, test_isolation):
        """Test isolating all resources."""
        with patch.object(test_isolation.db_isolation, "create_isolated_schema") as mock_db:
            mock_db.return_value = "test_schema"

            with patch.object(
                test_isolation.redis_isolation, "create_isolated_namespace"
            ) as mock_redis:
                mock_redis.return_value = "test:namespace"

                with patch.object(test_isolation.mq_isolation, "create_virtual_host") as mock_mq:
                    mock_mq.return_value = "test_vhost"

                    with patch.object(test_isolation.fs_isolation, "create_sandbox") as mock_fs:
                        mock_fs.return_value = Path("/tmp/sandbox")

                        context = test_isolation.isolate_all("test_suite")

                        assert context["database"]["schema"] == "test_schema"
                        assert context["redis"]["namespace"] == "test:namespace"
                        assert context["rabbitmq"]["vhost"] == "test_vhost"
                        assert context["filesystem"]["sandbox"] == Path("/tmp/sandbox")

    def test_cleanup_all(self, test_isolation):
        """Test cleaning up all resources."""
        context = {
            "database": {"schema": "test_schema"},
            "redis": {"namespace": "test:namespace"},
            "rabbitmq": {"vhost": "test_vhost"},
            "filesystem": {"sandbox": Path("/tmp/sandbox")},
        }

        with patch.object(test_isolation.db_isolation, "cleanup_schema") as mock_db:
            with patch.object(test_isolation.redis_isolation, "cleanup_namespace") as mock_redis:
                with patch.object(test_isolation.mq_isolation, "cleanup_virtual_host") as mock_mq:
                    with patch.object(test_isolation.fs_isolation, "cleanup_sandbox") as mock_fs:
                        test_isolation.cleanup_all(context)

                        mock_db.assert_called_with("test_schema")
                        mock_redis.assert_called_with("test:namespace")
                        mock_mq.assert_called_with("test_vhost")
                        mock_fs.assert_called_with(Path("/tmp/sandbox"))

    def test_context_manager(self, test_isolation):
        """Test using test isolation as context manager."""
        with patch.object(test_isolation, "isolate_all") as mock_isolate:
            mock_isolate.return_value = {"test": "context"}

            with patch.object(test_isolation, "cleanup_all") as mock_cleanup:
                with test_isolation.isolated("test") as context:
                    assert context == {"test": "context"}

                mock_cleanup.assert_called_with({"test": "context"})

    def test_get_isolation_level(self, test_isolation):
        """Test getting isolation level from config."""
        # Default level
        assert test_isolation.get_isolation_level() == IsolationLevel.SCHEMA

        # With config
        test_isolation.config["isolation_level"] = "DATABASE"
        assert test_isolation.get_isolation_level() == IsolationLevel.DATABASE
