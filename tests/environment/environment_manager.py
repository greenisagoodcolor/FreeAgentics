"""Environment Manager for test orchestration."""

import logging
import os
import subprocess
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import boto3
import docker
import pika
import psycopg2
import requests
from botocore.client import Config

import redis

logger = logging.getLogger(__name__)


class EnvironmentState(Enum):
    """State of the test environment."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    RESETTING = "resetting"


class PortAllocator:
    """Thread-safe port allocation for services."""

    def __init__(self, base_port: int = 50000):
        self.base_port = base_port
        self._allocated_ports: Dict[str, int] = {}
        self._used_ports: set = set()
        self._lock = threading.Lock()

    def allocate_port(self, service_name: str) -> int:
        """Allocate a port for a service."""
        with self._lock:
            # Return existing port if already allocated
            if service_name in self._allocated_ports:
                return self._allocated_ports[service_name]

            # Find next available port
            port = self.base_port
            while port in self._used_ports:
                port += 1

            self._allocated_ports[service_name] = port
            self._used_ports.add(port)

            return port

    def deallocate_port(self, service_name: str) -> None:
        """Free a port allocation."""
        with self._lock:
            if service_name in self._allocated_ports:
                port = self._allocated_ports[service_name]
                del self._allocated_ports[service_name]
                self._used_ports.remove(port)

    def get_allocated_ports(self) -> Dict[str, int]:
        """Get all allocated ports."""
        with self._lock:
            return self._allocated_ports.copy()

    def reset(self) -> None:
        """Reset all allocations."""
        with self._lock:
            self._allocated_ports.clear()
            self._used_ports.clear()


class ServiceHealth:
    """Health checking for various services."""

    def check_postgres(self, host: str, port: int, user: str, password: str, database: str) -> bool:
        """Check PostgreSQL health."""
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                connect_timeout=5,
            )
            conn.close()
            return True
        except Exception as e:
            logger.debug(f"PostgreSQL health check failed: {e}")
            return False

    def check_redis(self, host: str, port: int) -> bool:
        """Check Redis health."""
        try:
            client = redis.Redis(host=host, port=port, socket_connect_timeout=5)
            client.ping()
            client.close()
            return True
        except Exception as e:
            logger.debug(f"Redis health check failed: {e}")
            return False

    def check_rabbitmq(self, host: str, port: int, user: str, password: str, vhost: str) -> bool:
        """Check RabbitMQ health."""
        try:
            credentials = pika.PlainCredentials(user, password)
            parameters = pika.ConnectionParameters(
                host=host,
                port=port,
                virtual_host=vhost,
                credentials=credentials,
                connection_attempts=1,
                retry_delay=0,
            )
            connection = pika.BlockingConnection(parameters)
            connection.close()
            return True
        except Exception as e:
            logger.debug(f"RabbitMQ health check failed: {e}")
            return False

    def check_elasticsearch(self, url: str) -> bool:
        """Check Elasticsearch health."""
        try:
            response = requests.get(f"{url}/_cluster/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Elasticsearch health check failed: {e}")
            return False

    def check_minio(self, endpoint_url: str, access_key: str, secret_key: str) -> bool:
        """Check MinIO health."""
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(signature_version="s3v4"),
            )
            s3.list_buckets()
            return True
        except Exception as e:
            logger.debug(f"MinIO health check failed: {e}")
            return False

    def check_all_services(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """Check all services and return status."""
        results = {}

        if "postgres" in config:
            pg = config["postgres"]
            results["postgres"] = self.check_postgres(
                pg["host"],
                pg["port"],
                pg["user"],
                pg["password"],
                pg["database"],
            )

        if "redis" in config:
            rd = config["redis"]
            results["redis"] = self.check_redis(rd["host"], rd["port"])

        if "rabbitmq" in config:
            rmq = config["rabbitmq"]
            results["rabbitmq"] = self.check_rabbitmq(
                rmq["host"],
                rmq["port"],
                rmq["user"],
                rmq["password"],
                rmq["vhost"],
            )

        if "elasticsearch" in config:
            es = config["elasticsearch"]
            results["elasticsearch"] = self.check_elasticsearch(es["url"])

        if "minio" in config:
            minio = config["minio"]
            results["minio"] = self.check_minio(
                minio["endpoint_url"], minio["access_key"], minio["secret_key"]
            )

        return results


class EnvironmentManager:
    """Manages test environment lifecycle."""

    def __init__(
        self,
        compose_file: str = "docker-compose.test.yml",
        project_name: str = "freeagentics_test",
        work_dir: Optional[str] = None,
    ):
        self.compose_file = compose_file
        self.project_name = project_name
        self.work_dir = work_dir or os.getcwd()
        self.state = EnvironmentState.STOPPED
        self.port_allocator = PortAllocator()
        self.health_checker = ServiceHealth()
        self._docker_client = None

    @property
    def docker_client(self):
        """Lazy-load Docker client."""
        if self._docker_client is None:
            self._docker_client = docker.from_env()
        return self._docker_client

    def _run_compose_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run a docker-compose command."""
        full_command = [
            "docker-compose",
            "-f",
            self.compose_file,
            "-p",
            self.project_name,
        ] + command

        logger.debug(f"Running: {' '.join(full_command)}")

        return subprocess.run(full_command, cwd=self.work_dir, capture_output=True, text=True)

    def start(self, services: Optional[List[str]] = None) -> bool:
        """Start the test environment."""
        if self.state == EnvironmentState.RUNNING:
            logger.warning("Environment is already running")
            return True

        self.state = EnvironmentState.STARTING

        try:
            # Start services
            command = ["up", "-d"]
            if services:
                command.extend(services)

            result = self._run_compose_command(command)

            if result.returncode != 0:
                logger.error(f"Failed to start environment: {result.stderr}")
                self.state = EnvironmentState.ERROR
                return False

            # Wait for services to be healthy
            if self._wait_for_services():
                self.state = EnvironmentState.RUNNING
                logger.info("Test environment started successfully")
                return True
            else:
                logger.error("Services failed health checks")
                self.state = EnvironmentState.ERROR
                return False

        except Exception as e:
            logger.error(f"Error starting environment: {e}")
            self.state = EnvironmentState.ERROR
            return False

    def stop(self) -> bool:
        """Stop the test environment."""
        if self.state == EnvironmentState.STOPPED:
            logger.warning("Environment is already stopped")
            return True

        self.state = EnvironmentState.STOPPING

        try:
            result = self._run_compose_command(["down"])

            if result.returncode != 0:
                logger.error(f"Failed to stop environment: {result.stderr}")
                self.state = EnvironmentState.ERROR
                return False

            self.state = EnvironmentState.STOPPED
            logger.info("Test environment stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping environment: {e}")
            self.state = EnvironmentState.ERROR
            return False

    def reset(self) -> bool:
        """Reset the environment (stop, clean, start)."""
        self.state = EnvironmentState.RESETTING

        try:
            # Stop environment
            if not self.stop():
                return False

            # Clean volumes
            if not self._clean_volumes():
                return False

            # Start fresh
            return self.start()

        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            self.state = EnvironmentState.ERROR
            return False

    def _clean_volumes(self) -> bool:
        """Clean Docker volumes."""
        try:
            result = self._run_compose_command(["down", "-v"])
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error cleaning volumes: {e}")
            return False

    def _wait_for_services(self, timeout: int = 60) -> bool:
        """Wait for all services to be healthy."""
        start_time = time.time()
        config = self.get_service_config()

        while time.time() - start_time < timeout:
            health_status = self.health_checker.check_all_services(config)

            if all(health_status.values()):
                return True

            unhealthy = [k for k, v in health_status.items() if not v]
            logger.debug(f"Waiting for services: {unhealthy}")
            time.sleep(2)

        return False

    def get_service_config(self) -> Dict[str, Any]:
        """Get configuration for all services."""
        return {
            "postgres": {
                "host": "localhost",
                "port": self.port_allocator.allocate_port("postgres"),
                "user": "test_user",
                "password": "test_password",
                "database": "freeagentics_test",
            },
            "redis": {
                "host": "localhost",
                "port": self.port_allocator.allocate_port("redis"),
                "url": f"redis://localhost:{self.port_allocator.allocate_port('redis')}/0",
            },
            "rabbitmq": {
                "host": "localhost",
                "port": self.port_allocator.allocate_port("rabbitmq"),
                "user": "test_user",
                "password": "test_password",
                "vhost": "test_vhost",
            },
            "elasticsearch": {
                "host": "localhost",
                "port": self.port_allocator.allocate_port("elasticsearch"),
                "url": f"http://localhost:{self.port_allocator.allocate_port('elasticsearch')}",
            },
            "minio": {
                "host": "localhost",
                "port": self.port_allocator.allocate_port("minio"),
                "endpoint_url": f"http://localhost:{self.port_allocator.allocate_port('minio')}",
                "access_key": "test_access_key",
                "secret_key": "test_secret_key",
            },
        }

    def validate_state(self) -> bool:
        """Validate that environment is in expected state."""
        if self.state != EnvironmentState.RUNNING:
            return self.state == EnvironmentState.STOPPED

        config = self.get_service_config()
        health_status = self.health_checker.check_all_services(config)

        return all(health_status.values())

    def get_logs(self, service: str, lines: int = 100) -> str:
        """Get logs from a service."""
        result = self._run_compose_command(["logs", f"--tail={lines}", service])

        return result.stdout if result.returncode == 0 else result.stderr

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
