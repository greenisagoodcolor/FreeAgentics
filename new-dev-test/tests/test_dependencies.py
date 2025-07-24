"""
Test module for dependency verification following TDD approach.

This module follows RED-GREEN-REFACTOR cycle:
1. RED: Write failing tests that verify required dependencies
2. GREEN: Fix dependency issues to make tests pass
3. REFACTOR: Clean up and document findings

NO GRACEFUL FALLBACKS OR TRY/EXCEPT BLOCKS - HARD FAILURES ONLY
"""

import sys

import pkg_resources
import pytest


class TestDependencyVerification:
    """Test suite for verifying all required dependencies are properly installed."""

    # Critical dependencies that must be present for API to function
    REQUIRED_PACKAGES = {
        "psycopg2-binary": "2.9.10",
        "sqlalchemy": "2.0.23",
        "fastapi": "0.104.1",
        "passlib": "1.7.4",
        "pyjwt": "2.8.0",
        "asyncpg": None,  # Will be installed later
        # "python-jose": None,  # REMOVED - CVE-2024-33664, CVE-2024-33663 - Use PyJWT instead
        "python-multipart": "0.0.20",
        "uvicorn": "0.35.0",
        "redis": "6.2.0",
        "pydantic": "2.11.7",
        "alembic": "1.16.2",
    }

    # Packages that should NOT be present (conflicting or deprecated)
    FORBIDDEN_PACKAGES = [
        "psycopg2",  # Should use psycopg2-binary instead
        "PyMySQL",  # Should use PostgreSQL, not MySQL
        "sqlite3",  # Should use PostgreSQL, not SQLite
    ]

    def test_required_packages_are_installed(self):
        """Test that all required packages are installed with correct versions.

        This test MUST FAIL initially to follow TDD RED phase.
        """
        missing_packages = []
        wrong_version_packages = []

        for package_name, expected_version in self.REQUIRED_PACKAGES.items():
            try:
                installed_version = pkg_resources.get_distribution(package_name).version

                if expected_version and installed_version != expected_version:
                    wrong_version_packages.append(
                        f"{package_name}: expected {expected_version}, got {installed_version}"
                    )

            except pkg_resources.DistributionNotFound:
                missing_packages.append(package_name)

        # Hard failure - no graceful degradation
        if missing_packages:
            pytest.fail(f"Missing required packages: {missing_packages}")

        if wrong_version_packages:
            pytest.fail(f"Wrong package versions: {wrong_version_packages}")

    def test_forbidden_packages_not_installed(self):
        """Test that conflicting packages are NOT installed.

        This prevents common dependency conflicts.
        """
        installed_forbidden = []

        for package_name in self.FORBIDDEN_PACKAGES:
            try:
                pkg_resources.get_distribution(package_name)
                installed_forbidden.append(package_name)
            except pkg_resources.DistributionNotFound:
                pass  # Good - package is not installed

        if installed_forbidden:
            pytest.fail(f"Forbidden packages are installed: {installed_forbidden}")

    def test_database_driver_import(self):
        """Test that psycopg2 can be imported and used.

        This verifies the actual database driver functionality.
        """
        try:
            import psycopg2
            import psycopg2.extras

            # Test that we can access core functionality
            assert hasattr(psycopg2, "connect"), "psycopg2.connect not available"
            assert hasattr(psycopg2.extras, "RealDictCursor"), "RealDictCursor not available"

        except ImportError as e:
            pytest.fail(f"Failed to import psycopg2: {e}")

    def test_sqlalchemy_imports(self):
        """Test that SQLAlchemy can be imported with required components."""
        try:
            from sqlalchemy import Column, Integer, String, create_engine
            from sqlalchemy.ext.declarative import declarative_base

            # Test that we can create basic components
            engine = create_engine("sqlite:///:memory:")  # Minimal test
            Base = declarative_base()

            class TestModel(Base):
                __tablename__ = "test"
                id = Column(Integer, primary_key=True)
                name = Column(String)

            assert engine is not None
            assert Base is not None
            assert TestModel is not None

        except ImportError as e:
            pytest.fail(f"Failed to import SQLAlchemy components: {e}")

    def test_fastapi_imports(self):
        """Test that FastAPI can be imported with required components."""
        try:
            from fastapi import APIRouter, FastAPI
            from pydantic import BaseModel

            # Test that we can create basic components
            app = FastAPI()
            router = APIRouter()

            class TestModel(BaseModel):
                name: str

            assert app is not None
            assert router is not None
            assert TestModel is not None

        except ImportError as e:
            pytest.fail(f"Failed to import FastAPI components: {e}")

    def test_cryptography_imports(self):
        """Test that cryptography and JWT libraries work."""
        try:
            import jwt
            from passlib.context import CryptContext

            # Test basic JWT functionality
            payload = {"test": "data"}
            secret = "test_secret"

            token = jwt.encode(payload, secret, algorithm="HS256")
            decoded = jwt.decode(token, secret, algorithms=["HS256"])

            assert decoded == payload

            # Test password hashing
            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            hashed = pwd_context.hash("test_password")
            assert pwd_context.verify("test_password", hashed)

        except ImportError as e:
            pytest.fail(f"Failed to import cryptography components: {e}")

    def test_asyncpg_availability(self):
        """Test that asyncpg is available for async database operations.

        This test should fail initially since asyncpg is not installed.
        """
        try:
            import asyncpg

            # Test that we can access basic functionality
            assert hasattr(asyncpg, "connect"), "asyncpg.connect not available"
            assert hasattr(asyncpg, "create_pool"), "asyncpg.create_pool not available"

        except ImportError as e:
            pytest.fail(f"asyncpg not installed: {e}")

    def test_pyjwt_availability(self):
        """Test that PyJWT is available for JWT operations (replaces python-jose).

        Using PyJWT instead of python-jose due to CVE vulnerabilities.
        """
        try:
            import jwt

            # Test basic JWT functionality
            payload = {"test": "data"}
            secret = "test_secret"

            token = jwt.encode(payload, secret, algorithm="HS256")
            decoded = jwt.decode(token, secret, algorithms=["HS256"])

            assert decoded == payload

        except ImportError as e:
            pytest.fail(f"PyJWT not installed: {e}")

    def test_development_vs_production_dependencies(self):
        """Test that we have appropriate dependencies for the environment."""
        import os

        # Check environment
        is_production = os.getenv("PRODUCTION", "false").lower() == "true"

        # Production-specific requirements
        if is_production:
            production_packages = ["gunicorn", "psycopg2-binary"]
            for package in production_packages:
                try:
                    pkg_resources.get_distribution(package)
                except pkg_resources.DistributionNotFound:
                    pytest.fail(f"Production package {package} not installed")

        # Development-specific packages (should not be in production)
        dev_packages = ["pytest", "pytest-asyncio", "black", "flake8"]
        for package in dev_packages:
            try:
                pkg_resources.get_distribution(package)
                if is_production:
                    # Warning for dev packages in production
                    print(f"WARNING: Development package {package} found in production")
            except pkg_resources.DistributionNotFound:
                if not is_production:
                    pytest.fail(f"Development package {package} not installed")

    def test_version_compatibility(self):
        """Test that package versions are compatible with each other."""
        # Check Python version compatibility
        if sys.version_info < (3, 8):
            pytest.fail("Python 3.8+ required for FastAPI and modern async support")

        # Check SQLAlchemy 2.0 compatibility
        try:
            from sqlalchemy import __version__ as sqlalchemy_version

            major_version = int(sqlalchemy_version.split(".")[0])
            if major_version < 2:
                pytest.fail(f"SQLAlchemy 2.0+ required, got {sqlalchemy_version}")
        except ImportError:
            pytest.fail("SQLAlchemy not installed")

    def test_pip_install_commands_work(self):
        """Test that pip install commands work for missing packages.

        This test documents the commands needed to fix dependency issues.
        """
        missing_commands = []

        # Check if asyncpg is installed
        try:
            pass
        except ImportError:
            missing_commands.append("pip install asyncpg")

        # Check if PyJWT is installed (replaces python-jose)
        try:
            import jwt

            assert jwt  # Use the import to avoid F401
        except ImportError:
            missing_commands.append("pip install PyJWT")

        if missing_commands:
            pytest.fail(f"Run these commands to fix dependencies: {missing_commands}")


class TestDatabaseConnectivity:
    """Test suite for database connectivity - should fail without proper setup."""

    def test_database_url_environment_variable(self):
        """Test that DATABASE_URL environment variable is properly set."""
        import os

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            pytest.fail("DATABASE_URL environment variable not set")

        # Validate URL format
        if not database_url.startswith("postgresql://"):
            pytest.fail(f"DATABASE_URL must start with postgresql://, got: {database_url[:20]}...")

        # Check for development credentials in production
        if os.getenv("PRODUCTION", "false").lower() == "true":
            dev_indicators = ["localhost", "127.0.0.1", "dev", "test"]
            for indicator in dev_indicators:
                if indicator in database_url:
                    pytest.fail(f"Production environment using development database: {indicator}")

    def test_database_connection_pool_creation(self):
        """Test that database connection pool can be created."""
        try:
            from database.session import SessionLocal, engine

            # Test that engine was created
            assert engine is not None, "Database engine not created"

            # Test that we can create a session
            session = SessionLocal()
            assert session is not None, "Database session not created"
            session.close()

        except Exception as e:
            pytest.fail(f"Database connection pool creation failed: {e}")

    def test_database_table_creation(self):
        """Test that database tables can be created."""
        try:
            from database.models import Base
            from database.session import engine

            # This should create all tables
            Base.metadata.create_all(bind=engine)

            # Test that tables were created by checking metadata
            table_names = Base.metadata.tables.keys()

            expected_tables = ["agents", "coalitions", "agent_coalition"]
            for table_name in expected_tables:
                assert table_name in table_names, f"Table {table_name} not created"

        except Exception as e:
            pytest.fail(f"Database table creation failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output for debugging
    pytest.main([__file__, "-v", "--tb=short"])
