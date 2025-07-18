"""Database reset utilities for clean test runs."""

import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .pool_config import DatabasePool, close_all_pools, get_pool

logger = logging.getLogger(__name__)


class DatabaseReset:
    """Utilities for resetting database state between test runs."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        admin_user: str = "freeagentics",
        admin_password: str = "freeagentics123",
        admin_database: str = "postgres",
    ):
        """Initialize with database connection parameters."""
        self.host = host
        self.port = port
        self.admin_user = admin_user
        self.admin_password = admin_password
        self.admin_database = admin_database

        self.admin_url = f"postgresql://{admin_user}:{admin_password}@{host}:{port}/{admin_database}"

    def _get_admin_engine(self) -> Engine:
        """Get an engine with admin privileges."""
        # Create engine with autocommit for DDL operations
        engine = create_engine(
            self.admin_url, isolation_level="AUTOCOMMIT", pool_pre_ping=True
        )
        return engine

    def create_test_database(self, db_name: str = "freeagentics_test") -> bool:
        """Create a test database if it doesn't exist."""
        try:
            engine = self._get_admin_engine()

            with engine.connect() as conn:
                # Check if database exists
                result = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                    {"dbname": db_name},
                )

                if result.fetchone():
                    logger.info(f"Database '{db_name}' already exists")
                    return True

                # Create database
                conn.execute(
                    text(f"CREATE DATABASE {db_name} OWNER {self.admin_user}")
                )
                logger.info(f"Created database '{db_name}'")

            engine.dispose()
            return True

        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return False

    def drop_test_database(self, db_name: str = "freeagentics_test") -> bool:
        """Drop a test database."""
        try:
            # Close all existing connections
            close_all_pools()

            engine = self._get_admin_engine()

            with engine.connect() as conn:
                # Terminate existing connections
                conn.execute(
                    text(
                        """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = :dbname AND pid <> pg_backend_pid()
                """
                    ),
                    {"dbname": db_name},
                )

                # Drop database
                conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
                logger.info(f"Dropped database '{db_name}'")

            engine.dispose()
            return True

        except Exception as e:
            logger.error(f"Failed to drop database: {e}")
            return False

    def reset_database(self, db_name: str = "freeagentics_test") -> bool:
        """Reset database by dropping and recreating it."""
        logger.info(f"Resetting database '{db_name}'...")

        if not self.drop_test_database(db_name):
            return False

        time.sleep(1)  # Brief pause to ensure cleanup

        if not self.create_test_database(db_name):
            return False

        # Apply schema
        if not self.apply_schema(db_name):
            return False

        logger.info(f"Database '{db_name}' reset successfully")
        return True

    def apply_schema(self, db_name: str = "freeagentics_test") -> bool:
        """Apply the schema to a database."""
        schema_file = os.path.join(os.path.dirname(__file__), "schema.sql")

        if not os.path.exists(schema_file):
            logger.error(f"Schema file not found: {schema_file}")
            return False

        try:
            # Use psql to apply schema
            cmd = [
                "psql",
                "-h",
                self.host,
                "-p",
                str(self.port),
                "-U",
                self.admin_user,
                "-d",
                db_name,
                "-",
                schema_file,
            ]

            env = os.environ.copy()
            env["PGPASSWORD"] = self.admin_password

            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error(f"Schema application failed: {result.stderr}")
                return False

            logger.info(f"Schema applied to database '{db_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to apply schema: {e}")
            return False

    def truncate_all_tables(self, db_name: str = "freeagentics_test") -> bool:
        """Truncate all tables in the database."""
        try:
            pool = get_pool(
                "truncate",
                min_connections=1,
                max_connections=1,
                database=db_name,
            )

            with pool.get_session() as session:
                # Get all table names
                result = session.execute(
                    text(
                        """
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """
                    )
                )

                tables = [row.tablename for row in result]

                # Truncate all tables with CASCADE
                for table in tables:
                    session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                    logger.debug(f"Truncated table: {table}")

                logger.info(f"Truncated {len(tables)} tables")
                return True

        except Exception as e:
            logger.error(f"Failed to truncate tables: {e}")
            return False
        finally:
            close_all_pools()

    def create_snapshot(
        self,
        db_name: str = "freeagentics_test",
        snapshot_name: str = "test_snapshot",
    ) -> bool:
        """Create a database snapshot using pg_dump."""
        try:
            snapshot_file = f"/tmp/{snapshot_name}_{db_name}.sql"

            cmd = [
                "pg_dump",
                "-h",
                self.host,
                "-p",
                str(self.port),
                "-U",
                self.admin_user,
                "-d",
                db_name,
                "-",
                snapshot_file,
                "--no-owner",
                "--no-privileges",
            ]

            env = os.environ.copy()
            env["PGPASSWORD"] = self.admin_password

            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error(f"Snapshot creation failed: {result.stderr}")
                return False

            logger.info(f"Created snapshot: {snapshot_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return False

    def restore_snapshot(
        self,
        db_name: str = "freeagentics_test",
        snapshot_name: str = "test_snapshot",
    ) -> bool:
        """Restore database from a snapshot."""
        try:
            snapshot_file = f"/tmp/{snapshot_name}_{db_name}.sql"

            if not os.path.exists(snapshot_file):
                logger.error(f"Snapshot file not found: {snapshot_file}")
                return False

            # First reset the database
            if not self.reset_database(db_name):
                return False

            # Then restore from snapshot
            cmd = [
                "psql",
                "-h",
                self.host,
                "-p",
                str(self.port),
                "-U",
                self.admin_user,
                "-d",
                db_name,
                "-",
                snapshot_file,
            ]

            env = os.environ.copy()
            env["PGPASSWORD"] = self.admin_password

            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error(f"Snapshot restoration failed: {result.stderr}")
                return False

            logger.info(f"Restored database from snapshot: {snapshot_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            return False

    def get_table_counts(
        self, db_name: str = "freeagentics_test"
    ) -> Dict[str, int]:
        """Get row counts for all tables."""
        counts = {}

        try:
            pool = get_pool(
                "counts",
                min_connections=1,
                max_connections=1,
                database=db_name,
            )

            with pool.get_session() as session:
                # Get all table names
                result = session.execute(
                    text(
                        """
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """
                    )
                )

                tables = [row.tablename for row in result]

                # Get count for each table
                for table in tables:
                    result = session.execute(
                        text(f"SELECT COUNT(*) as count FROM {table}")
                    )
                    counts[table] = result.scalar()

            return counts

        except Exception as e:
            logger.error(f"Failed to get table counts: {e}")
            return counts
        finally:
            close_all_pools()

    def verify_schema(
        self, db_name: str = "freeagentics_test"
    ) -> Dict[str, Any]:
        """Verify that the schema is correctly applied."""
        verification = {
            "tables": {},
            "types": {},
            "indexes": {},
            "constraints": {},
            "triggers": {},
        }

        try:
            pool = get_pool(
                "verify",
                min_connections=1,
                max_connections=1,
                database=db_name,
            )

            with pool.get_session() as session:
                # Check tables
                result = session.execute(
                    text(
                        """
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                """
                    )
                )
                verification["tables"] = {
                    row.tablename: True for row in result
                }

                # Check custom types
                result = session.execute(
                    text(
                        """
                    SELECT typname
                    FROM pg_type
                    WHERE typnamespace = (
                        SELECT oid FROM pg_namespace WHERE nspname = 'public'
                    )
                    AND typtype = 'e'
                """
                    )
                )
                verification["types"] = {row.typname: True for row in result}

                # Check indexes
                result = session.execute(
                    text(
                        """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                """
                    )
                )
                verification["indexes"] = {
                    row.indexname: True for row in result
                }

                # Check constraints
                result = session.execute(
                    text(
                        """
                    SELECT conname
                    FROM pg_constraint
                    WHERE connamespace = (
                        SELECT oid FROM pg_namespace WHERE nspname = 'public'
                    )
                """
                    )
                )
                verification["constraints"] = {
                    row.conname: True for row in result
                }

                # Check triggers
                result = session.execute(
                    text(
                        """
                    SELECT tgname
                    FROM pg_trigger
                    WHERE tgrelid IN (
                        SELECT oid FROM pg_class
                        WHERE relnamespace = (
                            SELECT oid FROM pg_namespace WHERE nspname = 'public'
                        )
                    )
                    AND tgisinternal = false
                """
                    )
                )
                verification["triggers"] = {row.tgname: True for row in result}

            return verification

        except Exception as e:
            logger.error(f"Failed to verify schema: {e}")
            return verification
        finally:
            close_all_pools()


def main():
    """Command-line interface for database reset operations."""
    import argparse

    parser = argparse.ArgumentParser(description="Database reset utilities")
    parser.add_argument(
        "command",
        choices=[
            "reset",
            "create",
            "drop",
            "truncate",
            "snapshot",
            "restore",
            "verify",
            "counts",
        ],
    )
    parser.add_argument(
        "--database", default="freeagentics_test", help="Database name"
    )
    parser.add_argument(
        "--snapshot",
        default="test_snapshot",
        help="Snapshot name for backup/restore",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--user", default="freeagentics")
    parser.add_argument("--password", default="freeagentics123")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create reset utility
    reset_util = DatabaseReset(
        host=args.host,
        port=args.port,
        admin_user=args.user,
        admin_password=args.password,
    )

    # Execute command
    if args.command == "reset":
        success = reset_util.reset_database(args.database)
    elif args.command == "create":
        success = reset_util.create_test_database(args.database)
    elif args.command == "drop":
        success = reset_util.drop_test_database(args.database)
    elif args.command == "truncate":
        success = reset_util.truncate_all_tables(args.database)
    elif args.command == "snapshot":
        success = reset_util.create_snapshot(args.database, args.snapshot)
    elif args.command == "restore":
        success = reset_util.restore_snapshot(args.database, args.snapshot)
    elif args.command == "verify":
        result = reset_util.verify_schema(args.database)
        print("\nSchema Verification:")
        for category, items in result.items():
            print(f"\n{category.upper()}:")
            for name, exists in items.items():
                print(f"  - {name}: {'✓' if exists else '✗'}")
        success = True
    elif args.command == "counts":
        counts = reset_util.get_table_counts(args.database)
        print("\nTable Row Counts:")
        for table, count in counts.items():
            print(f"  - {table}: {count}")
        success = True

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
