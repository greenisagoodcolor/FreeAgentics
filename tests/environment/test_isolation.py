"""Test isolation framework for creating isolated test environments."""

import logging
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pika
import psycopg2

import redis

logger = logging.getLogger(__name__)


class IsolationLevel(Enum):
    """Levels of test isolation."""

    SCHEMA = "schema"  # Isolated database schema
    DATABASE = "database"  # Isolated database
    CONTAINER = "container"  # Isolated container
    PROCESS = "process"  # Isolated process


class DatabaseIsolation:
    """Database isolation for PostgreSQL tests."""

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        """Initialize database isolation manager.

        Args:
            host: Database host
            port: Database port
            user: Database username
            password: Database password
            database: Database name
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._schemas: List[str] = []
        self._databases: List[str] = []

    def create_isolated_schema(self, prefix: str) -> str:
        """Create an isolated schema for testing."""
        schema_name = f"{prefix}_{uuid.uuid4().hex[:8]}"

        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
        )

        try:
            cursor = conn.cursor()
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
            cursor.execute(f"SET search_path TO {schema_name}")
            conn.commit()
            self._schemas.append(schema_name)
            return schema_name
        finally:
            conn.close()

    def create_isolated_database(self, prefix: str) -> str:
        """Create an isolated database for testing."""
        db_name = f"{prefix}_{uuid.uuid4().hex[:8]}"

        # Connect to postgres database to create new database
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database="postgres",
        )
        conn.autocommit = True

        try:
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE {db_name}")
            self._databases.append(db_name)
            return db_name
        finally:
            conn.close()

    def cleanup_schema(self, schema_name: str) -> None:
        """Clean up an isolated schema."""
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
        )

        try:
            cursor = conn.cursor()
            cursor.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
            conn.commit()
            if schema_name in self._schemas:
                self._schemas.remove(schema_name)
        finally:
            conn.close()

    def cleanup_database(self, db_name: str) -> None:
        """Clean up an isolated database."""
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database="postgres",
        )
        conn.autocommit = True

        try:
            cursor = conn.cursor()

            # Terminate existing connections
            cursor.execute(
                """
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = %s
                AND pid <> pg_backend_pid()
            """,
                (db_name,),
            )

            # Drop database
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")

            if db_name in self._databases:
                self._databases.remove(db_name)
        finally:
            conn.close()

    @contextmanager
    def isolated_schema(self, prefix: str):
        """Context manager for isolated schema."""
        schema_name = self.create_isolated_schema(prefix)
        try:
            yield schema_name
        finally:
            self.cleanup_schema(schema_name)

    @contextmanager
    def isolated_database(self, prefix: str):
        """Context manager for isolated database."""
        db_name = self.create_isolated_database(prefix)
        try:
            yield db_name
        finally:
            self.cleanup_database(db_name)

    def get_connection_url(self, schema: Optional[str] = None) -> str:
        """Get database connection URL."""
        base_url = (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

        if schema:
            base_url += f"?options=--search_path%3D{schema}"

        return base_url

    def cleanup_all(self) -> None:
        """Clean up all created resources."""
        for schema in self._schemas[:]:
            self.cleanup_schema(schema)
        for db in self._databases[:]:
            self.cleanup_database(db)


class RedisNamespacedClient:
    """Redis client with namespace support."""

    def __init__(self, client: redis.Redis, namespace: str):
        """Initialize namespaced Redis client.

        Args:
            client: Redis client instance
            namespace: Namespace prefix for keys
        """
        self.client = client
        self.namespace = namespace

    def _key(self, key: str) -> str:
        """Add namespace prefix to key."""
        return f"{self.namespace}:{key}"

    def get(self, key: str):
        """Get value with namespace."""
        return self.client.get(self._key(key))

    def set(self, key: str, value: Any, **kwargs):
        """Set value with namespace."""
        return self.client.set(self._key(key), value, **kwargs)

    def delete(self, key: str):
        """Delete key with namespace."""
        return self.client.delete(self._key(key))


class RedisIsolation:
    """Redis isolation for testing."""

    def __init__(self, host: str, port: int, db: int = 0):
        """Initialize Redis isolation manager.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
        """
        self.host = host
        self.port = port
        self.db = db
        self._namespaces: List[str] = []

    def create_isolated_namespace(self, prefix: str) -> str:
        """Create an isolated namespace."""
        namespace = f"test:{prefix}:{uuid.uuid4().hex[:8]}"
        self._namespaces.append(namespace)
        return namespace

    def get_namespaced_client(self, namespace: str) -> RedisNamespacedClient:
        """Get a namespaced Redis client."""
        client = redis.Redis(host=self.host, port=self.port, db=self.db)
        return RedisNamespacedClient(client, namespace)

    def cleanup_namespace(self, namespace: str) -> None:
        """Clean up all keys in a namespace."""
        client = redis.Redis(host=self.host, port=self.port, db=self.db)

        keys = list(client.scan_iter(f"{namespace}:*"))
        if keys:
            client.delete(*keys)

        if namespace in self._namespaces:
            self._namespaces.remove(namespace)

    @contextmanager
    def isolated_namespace(self, prefix: str):
        """Context manager for isolated namespace."""
        namespace = self.create_isolated_namespace(prefix)
        try:
            yield namespace
        finally:
            self.cleanup_namespace(namespace)

    def cleanup_all(self) -> None:
        """Clean up all created namespaces."""
        for namespace in self._namespaces[:]:
            self.cleanup_namespace(namespace)


class MessageQueueIsolation:
    """Message queue isolation for RabbitMQ tests."""

    def __init__(self, host: str, port: int, user: str, password: str):
        """Initialize message queue isolation manager.

        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            user: RabbitMQ username
            password: RabbitMQ password
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self._vhosts: List[str] = []

    def create_virtual_host(self, prefix: str) -> str:
        """Create an isolated virtual host."""
        vhost = f"{prefix}_{uuid.uuid4().hex[:8]}"

        # Use RabbitMQ management API
        import requests

        url = f"http://{self.host}:15672/api/vhosts/{vhost}"
        response = requests.put(url, auth=(self.user, self.password))

        if response.status_code in [201, 204]:
            self._vhosts.append(vhost)
            return vhost
        else:
            raise Exception(f"Failed to create virtual host: {response.status_code}")

    def cleanup_virtual_host(self, vhost: str) -> None:
        """Clean up a virtual host."""
        import requests

        url = f"http://{self.host}:15672/api/vhosts/{vhost}"
        response = requests.delete(url, auth=(self.user, self.password))

        if response.status_code in [204, 404]:
            if vhost in self._vhosts:
                self._vhosts.remove(vhost)
        else:
            logger.warning(f"Failed to delete virtual host {vhost}: {response.status_code}")

    def get_connection_params(self, vhost: str) -> pika.ConnectionParameters:
        """Get connection parameters for a virtual host."""
        credentials = pika.PlainCredentials(self.user, self.password)
        return pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=vhost,
            credentials=credentials,
        )

    @contextmanager
    def isolated_vhost(self, prefix: str):
        """Context manager for isolated virtual host."""
        vhost = self.create_virtual_host(prefix)
        try:
            yield vhost
        finally:
            self.cleanup_virtual_host(vhost)

    def cleanup_all(self) -> None:
        """Clean up all created virtual hosts."""
        for vhost in self._vhosts[:]:
            self.cleanup_virtual_host(vhost)


class FilesystemIsolation:
    """Filesystem isolation for testing."""

    def __init__(self, base_dir: str = "/tmp/test_isolation"):
        """Initialize filesystem isolation manager.

        Args:
            base_dir: Base directory for creating sandboxes
        """
        self.base_dir = base_dir
        self._sandboxes: List[Path] = []

        # Ensure base directory exists
        Path(base_dir).mkdir(parents=True, exist_ok=True)

    def create_sandbox(self, prefix: str) -> Path:
        """Create an isolated filesystem sandbox."""
        sandbox = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=self.base_dir))
        self._sandboxes.append(sandbox)
        return sandbox

    def cleanup_sandbox(self, sandbox: Union[str, Path]) -> None:
        """Clean up a sandbox."""
        sandbox_path = Path(sandbox)

        if sandbox_path.exists():
            shutil.rmtree(sandbox_path, ignore_errors=True)

        if sandbox_path in self._sandboxes:
            self._sandboxes.remove(sandbox_path)

    def copy_to_sandbox(
        self,
        source: Union[str, Path],
        sandbox: Path,
        dest_name: Optional[str] = None,
    ) -> Path:
        """Copy a file or directory to the sandbox."""
        source_path = Path(source)
        dest_name = dest_name or source_path.name
        dest_path = sandbox / dest_name

        if source_path.is_dir():
            shutil.copytree(source_path, dest_path)
        else:
            shutil.copy2(source_path, dest_path)

        return dest_path

    @contextmanager
    def sandboxed(self, prefix: str):
        """Context manager for sandboxed filesystem."""
        sandbox = self.create_sandbox(prefix)
        try:
            yield sandbox
        finally:
            self.cleanup_sandbox(sandbox)

    def cleanup_all(self) -> None:
        """Clean up all created sandboxes."""
        for sandbox in self._sandboxes[:]:
            self.cleanup_sandbox(sandbox)


class IsolationTester:
    """Main test isolation coordinator."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize test isolation coordinator.

        Args:
            config: Configuration dictionary for various isolation backends
        """
        self.config = config

        # Initialize isolation components
        if "postgres" in config:
            pg_config = config["postgres"]
            self.db_isolation = DatabaseIsolation(
                host=pg_config["host"],
                port=pg_config["port"],
                user=pg_config["user"],
                password=pg_config["password"],
                database=pg_config["database"],
            )
        else:
            self.db_isolation = None

        if "redis" in config:
            redis_config = config["redis"]
            self.redis_isolation = RedisIsolation(
                host=redis_config["host"],
                port=redis_config["port"],
                db=redis_config.get("db", 0),
            )
        else:
            self.redis_isolation = None

        if "rabbitmq" in config:
            mq_config = config["rabbitmq"]
            self.mq_isolation = MessageQueueIsolation(
                host=mq_config["host"],
                port=mq_config["port"],
                user=mq_config["user"],
                password=mq_config["password"],
            )
        else:
            self.mq_isolation = None

        self.fs_isolation = FilesystemIsolation(
            base_dir=config.get("filesystem_base", "/tmp/test_isolation")
        )

    def get_isolation_level(self) -> IsolationLevel:
        """Get the configured isolation level."""
        level_str = self.config.get("isolation_level", "SCHEMA")
        # Convert to lowercase to match enum values
        return IsolationLevel(level_str.lower())

    def isolate_all(self, prefix: str) -> Dict[str, Any]:
        """Create isolation for all configured resources."""
        context = {}

        if self.db_isolation:
            level = self.get_isolation_level()
            if level == IsolationLevel.DATABASE:
                db_name = self.db_isolation.create_isolated_database(prefix)
                context["database"] = {"database": db_name}
            else:
                schema_name = self.db_isolation.create_isolated_schema(prefix)
                context["database"] = {"schema": schema_name}

        if self.redis_isolation:
            namespace = self.redis_isolation.create_isolated_namespace(prefix)
            context["redis"] = {"namespace": namespace}

        if self.mq_isolation:
            vhost = self.mq_isolation.create_virtual_host(prefix)
            context["rabbitmq"] = {"vhost": vhost}

        sandbox = self.fs_isolation.create_sandbox(prefix)
        context["filesystem"] = {"sandbox": sandbox}

        return context

    def cleanup_all(self, context: Dict[str, Any]) -> None:
        """Clean up all isolated resources."""
        if "database" in context:
            db_context = context["database"]
            if "schema" in db_context:
                self.db_isolation.cleanup_schema(db_context["schema"])
            elif "database" in db_context:
                self.db_isolation.cleanup_database(db_context["database"])

        if "redis" in context:
            self.redis_isolation.cleanup_namespace(context["redis"]["namespace"])

        if "rabbitmq" in context:
            self.mq_isolation.cleanup_virtual_host(context["rabbitmq"]["vhost"])

        if "filesystem" in context:
            self.fs_isolation.cleanup_sandbox(context["filesystem"]["sandbox"])

    @contextmanager
    def isolated(self, prefix: str):
        """Context manager for complete isolation."""
        context = self.isolate_all(prefix)
        try:
            yield context
        finally:
            self.cleanup_all(context)

    def cleanup_orphaned_resources(self) -> None:
        """Clean up any orphaned test resources."""
        if self.db_isolation:
            self.db_isolation.cleanup_all()
        if self.redis_isolation:
            self.redis_isolation.cleanup_all()
        if self.mq_isolation:
            self.mq_isolation.cleanup_all()
        self.fs_isolation.cleanup_all()
