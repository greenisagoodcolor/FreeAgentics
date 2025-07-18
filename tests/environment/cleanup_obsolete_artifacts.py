#!/usr/bin/env python3
"""
Cleanup script for obsolete test environment artifacts.
This script removes old Docker containers, volumes, networks, and isolation resources.
"""

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import docker
import psycopg2

import redis

# Add project root to path
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)


@dataclass
class CleanupStats:
    """Statistics for cleanup operations."""

    containers_removed: int = 0
    volumes_removed: int = 0
    networks_removed: int = 0
    images_removed: int = 0
    schemas_removed: int = 0
    redis_keys_removed: int = 0
    rabbitmq_vhosts_removed: int = 0
    filesystem_artifacts_removed: int = 0
    total_size_freed: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ObsoleteArtifactCleaner:
    """Cleaner for obsolete test environment artifacts."""

    def __init__(self, dry_run: bool = False, max_age_hours: int = 24):
        self.dry_run = dry_run
        self.max_age_hours = max_age_hours
        self.cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        self.docker_client = docker.from_env()
        self.stats = CleanupStats()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def cleanup_docker_containers(self, patterns: List[str]) -> int:
        """Clean up Docker containers matching patterns."""
        removed_count = 0

        try:
            containers = self.docker_client.containers.list(all=True)

            for container in containers:
                if self._matches_patterns(container.name, patterns):
                    # Check if container is old enough
                    created_time = datetime.fromisoformat(
                        container.attrs["Created"].replace("Z", "+00:00")
                    )

                    if created_time < self.cutoff_time:
                        if not self.dry_run:
                            try:
                                if container.status == "running":
                                    container.stop(timeout=10)
                                container.remove()
                                removed_count += 1
                                self.logger.info(
                                    f"Removed container: {container.name}"
                                )
                            except Exception as e:
                                self.stats.errors.append(
                                    f"Error removing container {container.name}: {e}"
                                )
                        else:
                            self.logger.info(
                                f"Would remove container: {container.name}"
                            )
                            removed_count += 1

        except Exception as e:
            self.stats.errors.append(f"Error listing containers: {e}")

        return removed_count

    def cleanup_docker_volumes(self, patterns: List[str]) -> int:
        """Clean up Docker volumes matching patterns."""
        removed_count = 0
        size_freed = 0

        try:
            volumes = self.docker_client.volumes.list()

            for volume in volumes:
                if self._matches_patterns(volume.name, patterns):
                    # Check if volume is old enough
                    created_time = datetime.fromisoformat(
                        volume.attrs["CreatedAt"].replace("Z", "+00:00")
                    )

                    if created_time < self.cutoff_time:
                        if not self.dry_run:
                            try:
                                # Get size before removal
                                size = self._get_volume_size(volume)
                                volume.remove()
                                removed_count += 1
                                size_freed += size
                                self.logger.info(
                                    f"Removed volume: {volume.name} ({size} bytes)"
                                )
                            except Exception as e:
                                self.stats.errors.append(
                                    f"Error removing volume {volume.name}: {e}"
                                )
                        else:
                            self.logger.info(
                                f"Would remove volume: {volume.name}"
                            )
                            removed_count += 1

        except Exception as e:
            self.stats.errors.append(f"Error listing volumes: {e}")

        return removed_count

    def cleanup_docker_networks(self, patterns: List[str]) -> int:
        """Clean up Docker networks matching patterns."""
        removed_count = 0

        try:
            networks = self.docker_client.networks.list()

            for network in networks:
                if self._matches_patterns(
                    network.name, patterns
                ) and network.name not in [
                    "bridge",
                    "host",
                    "none",
                ]:
                    # Check if network is old enough
                    created_time = datetime.fromisoformat(
                        network.attrs["Created"].replace("Z", "+00:00")
                    )

                    if created_time < self.cutoff_time:
                        if not self.dry_run:
                            try:
                                network.remove()
                                removed_count += 1
                                self.logger.info(
                                    f"Removed network: {network.name}"
                                )
                            except Exception as e:
                                self.stats.errors.append(
                                    f"Error removing network {network.name}: {e}"
                                )
                        else:
                            self.logger.info(
                                f"Would remove network: {network.name}"
                            )
                            removed_count += 1

        except Exception as e:
            self.stats.errors.append(f"Error listing networks: {e}")

        return removed_count

    def cleanup_docker_images(self, patterns: List[str]) -> int:
        """Clean up Docker images matching patterns."""
        removed_count = 0

        try:
            images = self.docker_client.images.list()

            for image in images:
                tags = image.tags or []

                for tag in tags:
                    if self._matches_patterns(tag, patterns):
                        # Check if image is old enough
                        created_time = datetime.fromtimestamp(
                            image.attrs["Created"]
                        )

                        if created_time < self.cutoff_time:
                            if not self.dry_run:
                                try:
                                    self.docker_client.images.remove(
                                        image.id, force=True
                                    )
                                    removed_count += 1
                                    self.logger.info(f"Removed image: {tag}")
                                    break
                                except Exception as e:
                                    self.stats.errors.append(
                                        f"Error removing image {tag}: {e}"
                                    )
                            else:
                                self.logger.info(f"Would remove image: {tag}")
                                removed_count += 1
                                break

        except Exception as e:
            self.stats.errors.append(f"Error listing images: {e}")

        return removed_count

    def cleanup_database_schemas(self) -> int:
        """Clean up obsolete database schemas."""
        removed_count = 0

        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5433,
                user="test_user",
                password="test_password",
                database="freeagentics_test",
            )

            cursor = conn.cursor()

            # Find test schemas
            cursor.execute(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name LIKE 'test_%'
                AND schema_name NOT IN ('information_schema', 'pg_catalog')
            """
            )

            schemas = cursor.fetchall()

            for (schema_name,) in schemas:
                if not self.dry_run:
                    try:
                        cursor.execute(
                            f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"
                        )
                        conn.commit()
                        removed_count += 1
                        self.logger.info(f"Removed schema: {schema_name}")
                    except Exception as e:
                        self.stats.errors.append(
                            f"Error removing schema {schema_name}: {e}"
                        )
                        conn.rollback()
                else:
                    self.logger.info(f"Would remove schema: {schema_name}")
                    removed_count += 1

            conn.close()

        except Exception as e:
            self.stats.errors.append(f"Error connecting to database: {e}")

        return removed_count

    def cleanup_redis_namespaces(self) -> int:
        """Clean up Redis test namespaces."""
        removed_count = 0

        try:
            client = redis.Redis(host="localhost", port=6380, db=0)

            # Find test keys
            test_keys = list(client.scan_iter("test:*"))

            if test_keys:
                if not self.dry_run:
                    try:
                        client.delete(*test_keys)
                        removed_count = len(test_keys)
                        self.logger.info(
                            f"Removed {removed_count} Redis test keys"
                        )
                    except Exception as e:
                        self.stats.errors.append(
                            f"Error removing Redis keys: {e}"
                        )
                else:
                    self.logger.info(
                        f"Would remove {len(test_keys)} Redis test keys"
                    )
                    removed_count = len(test_keys)

        except Exception as e:
            self.stats.errors.append(f"Error connecting to Redis: {e}")

        return removed_count

    def cleanup_rabbitmq_vhosts(self) -> int:
        """Clean up RabbitMQ test virtual hosts."""
        removed_count = 0

        try:
            import requests

            # List virtual hosts
            response = requests.get(
                "http://localhost:15673/api/vhosts",
                auth=("test_user", "test_password"),
                timeout=10,
            )

            if response.status_code == 200:
                vhosts = response.json()

                for vhost in vhosts:
                    vhost_name = vhost["name"]
                    if vhost_name.startswith("test_"):
                        if not self.dry_run:
                            try:
                                delete_response = requests.delete(
                                    f"http://localhost:15673/api/vhosts/{vhost_name}",
                                    auth=("test_user", "test_password"),
                                    timeout=10,
                                )

                                if delete_response.status_code in [204, 404]:
                                    removed_count += 1
                                    self.logger.info(
                                        f"Removed RabbitMQ vhost: {vhost_name}"
                                    )
                                else:
                                    self.stats.errors.append(
                                        f"Error removing vhost {vhost_name}: HTTP {delete_response.status_code}"
                                    )
                            except Exception as e:
                                self.stats.errors.append(
                                    f"Error removing vhost {vhost_name}: {e}"
                                )
                        else:
                            self.logger.info(
                                f"Would remove RabbitMQ vhost: {vhost_name}"
                            )
                            removed_count += 1

        except Exception as e:
            self.stats.errors.append(f"Error connecting to RabbitMQ: {e}")

        return removed_count

    def cleanup_filesystem_artifacts(self) -> int:
        """Clean up filesystem test artifacts."""
        removed_count = 0

        test_dirs = [
            Path("/tmp/test_isolation"),
            Path("/tmp/pytest-*"),
            Path("/tmp/test_*"),
            Path(".pytest_cache"),
            Path("htmlcov"),
            Path("test-reports"),
            Path("*.pyc"),
            Path("__pycache__"),
        ]

        for pattern in test_dirs:
            if pattern.is_absolute():
                # Absolute path
                if pattern.exists():
                    if not self.dry_run:
                        try:
                            if pattern.is_dir():
                                shutil.rmtree(pattern)
                            else:
                                pattern.unlink()
                            removed_count += 1
                            self.logger.info(
                                f"Removed filesystem artifact: {pattern}"
                            )
                        except Exception as e:
                            self.stats.errors.append(
                                f"Error removing {pattern}: {e}"
                            )
                    else:
                        self.logger.info(
                            f"Would remove filesystem artifact: {pattern}"
                        )
                        removed_count += 1
            else:
                # Glob pattern
                for path in Path(".").glob(str(pattern)):
                    if self._is_old_enough(path):
                        if not self.dry_run:
                            try:
                                if path.is_dir():
                                    shutil.rmtree(path)
                                else:
                                    path.unlink()
                                removed_count += 1
                                self.logger.info(
                                    f"Removed filesystem artifact: {path}"
                                )
                            except Exception as e:
                                self.stats.errors.append(
                                    f"Error removing {path}: {e}"
                                )
                        else:
                            self.logger.info(
                                f"Would remove filesystem artifact: {path}"
                            )
                            removed_count += 1

        return removed_count

    def _matches_patterns(self, name: str, patterns: List[str]) -> bool:
        """Check if name matches any of the patterns."""
        for pattern in patterns:
            if "*" in pattern:
                # Simple wildcard matching
                if pattern.startswith("*") and name.endswith(pattern[1:]):
                    return True
                elif pattern.endswith("*") and name.startswith(pattern[:-1]):
                    return True
                elif "*" in pattern.replace("*", ""):
                    # More complex pattern
                    import fnmatch

                    return fnmatch.fnmatch(name, pattern)
            else:
                if name == pattern:
                    return True
        return False

    def _get_volume_size(self, volume) -> int:
        """Get volume size in bytes."""
        try:
            # This is a simplified approach
            # In reality, you'd need to inspect the volume's mount point
            return 0
        except:
            return 0

    def _is_old_enough(self, path: Path) -> bool:
        """Check if path is old enough to be cleaned up."""
        try:
            stat = path.stat()
            created_time = datetime.fromtimestamp(stat.st_ctime)
            return created_time < self.cutoff_time
        except:
            return False

    def run_cleanup(self) -> CleanupStats:
        """Run complete cleanup process."""
        self.logger.info(
            f"Starting cleanup (dry_run={self.dry_run}, max_age_hours={self.max_age_hours})"
        )

        # Docker cleanup
        self.stats.containers_removed = self.cleanup_docker_containers(
            ["freeagentics-test-*", "test_*", "*_test_*"]
        )

        self.stats.volumes_removed = self.cleanup_docker_volumes(
            ["test-*", "*_test_*"]
        )

        self.stats.networks_removed = self.cleanup_docker_networks(
            ["test-*", "*_test_*"]
        )

        self.stats.images_removed = self.cleanup_docker_images(
            ["test-*", "*:test-*"]
        )

        # Database cleanup
        self.stats.schemas_removed = self.cleanup_database_schemas()

        # Redis cleanup
        self.stats.redis_keys_removed = self.cleanup_redis_namespaces()

        # RabbitMQ cleanup
        self.stats.rabbitmq_vhosts_removed = self.cleanup_rabbitmq_vhosts()

        # Filesystem cleanup
        self.stats.filesystem_artifacts_removed = (
            self.cleanup_filesystem_artifacts()
        )

        self.logger.info("Cleanup completed")
        self._print_summary()

        return self.stats

    def _print_summary(self):
        """Print cleanup summary."""
        print("\n" + "=" * 60)
        print("CLEANUP SUMMARY")
        print("=" * 60)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'REAL CLEANUP'}")
        print(f"Max age: {self.max_age_hours} hours")
        print(f"Cutoff time: {self.cutoff_time}")
        print("-" * 60)
        print(f"Containers removed: {self.stats.containers_removed}")
        print(f"Volumes removed: {self.stats.volumes_removed}")
        print(f"Networks removed: {self.stats.networks_removed}")
        print(f"Images removed: {self.stats.images_removed}")
        print(f"Database schemas removed: {self.stats.schemas_removed}")
        print(f"Redis keys removed: {self.stats.redis_keys_removed}")
        print(f"RabbitMQ vhosts removed: {self.stats.rabbitmq_vhosts_removed}")
        print(
            f"Filesystem artifacts removed: {self.stats.filesystem_artifacts_removed}"
        )
        print(f"Total errors: {len(self.stats.errors)}")

        if self.stats.errors:
            print("\nErrors:")
            for error in self.stats.errors:
                print(f"  - {error}")

        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up obsolete test environment artifacts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned up without actually doing it",
    )
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=24,
        help="Maximum age in hours for artifacts to be considered for cleanup",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force cleanup without confirmation",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.dry_run and not args.force:
        response = input(
            f"This will clean up test artifacts older than {args.max_age_hours} hours. "
            "Continue? (y/N): "
        )
        if response.lower() != "y":
            print("Cleanup cancelled.")
            return

    cleaner = ObsoleteArtifactCleaner(
        dry_run=args.dry_run, max_age_hours=args.max_age_hours
    )

    stats = cleaner.run_cleanup()

    # Exit with error code if there were errors
    if stats.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
