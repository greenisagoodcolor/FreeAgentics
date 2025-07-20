"""Comprehensive tests for database.session module to achieve high coverage."""

import os
import warnings
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session


class TestDatabaseSession:
    """Test database session management."""

    def setup_method(self):
        """Setup test environment."""
        # Store original environment
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Restore original environment."""
        # Restore environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_development_mode_detection(self):
        """Test development mode detection from various env vars."""
        # Test DEVELOPMENT_MODE
        os.environ.clear()
        os.environ["DEVELOPMENT_MODE"] = "true"
        os.environ[
            "DATABASE_URL"
        ] = "sqlite:///:memory:"  # Add URL to prevent error

        # Import module to test
        import importlib

        import database.session

        importlib.reload(database.session)

        assert database.session.is_development is True

        # Test ENVIRONMENT
        os.environ.clear()
        os.environ["ENVIRONMENT"] = "development"
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        importlib.reload(database.session)
        assert database.session.is_development is True

        # Test ENV
        os.environ.clear()
        os.environ["ENV"] = "development"
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        importlib.reload(database.session)
        assert database.session.is_development is True

        # Test production
        os.environ.clear()
        os.environ["DEVELOPMENT_MODE"] = "false"
        os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost/db"
        importlib.reload(database.session)
        assert database.session.is_development is False

    def test_database_url_fallback_development(self):
        """Test SQLite fallback in development mode."""
        os.environ.clear()
        os.environ["DEVELOPMENT_MODE"] = "true"

        # Should use SQLite fallback and warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib

            import database.session

            importlib.reload(database.session)

            assert len(w) == 1
            assert "Using SQLite for development" in str(w[0].message)
            assert (
                database.session.DATABASE_URL
                == "sqlite:///./freeagentics_dev.db"
            )

    def test_database_url_required_production(self):
        """Test DATABASE_URL is required in production."""
        os.environ.clear()
        os.environ["DEVELOPMENT_MODE"] = "false"

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            import importlib

            import database.session

            importlib.reload(database.session)

        assert "DATABASE_URL environment variable is required" in str(
            exc_info.value
        )

    def test_production_security_validation(self):
        """Test production security validations."""
        os.environ.clear()
        os.environ["PRODUCTION"] = "true"
        os.environ[
            "DATABASE_URL"
        ] = "postgresql://freeagentics_dev_2025:pass@localhost/db"

        # Should reject dev credentials in production
        with pytest.raises(ValueError) as exc_info:
            import importlib

            import database.session

            importlib.reload(database.session)

        assert "using development database credentials" in str(exc_info.value)

        # Test with freeagentics123
        os.environ[
            "DATABASE_URL"
        ] = "postgresql://user:freeagentics123@localhost/db"
        with pytest.raises(ValueError):
            importlib.reload(database.session)

    def test_production_ssl_requirement(self):
        """Test SSL is added in production."""
        os.environ.clear()
        os.environ["PRODUCTION"] = "true"
        os.environ[
            "DATABASE_URL"
        ] = "postgresql://user:secure_pass@localhost/db"

        import importlib

        import database.session

        importlib.reload(database.session)

        assert "sslmode=require" in database.session.DATABASE_URL

    def test_engine_configuration_postgresql(self):
        """Test PostgreSQL engine configuration."""
        os.environ.clear()
        os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost/db"
        os.environ["DB_POOL_SIZE"] = "30"
        os.environ["DB_MAX_OVERFLOW"] = "50"
        os.environ["DB_POOL_TIMEOUT"] = "45"
        os.environ["DEBUG_SQL"] = "true"

        with patch('sqlalchemy.create_engine') as mock_create_engine:
            import importlib

            import database.session

            importlib.reload(database.session)

            mock_create_engine.assert_called_once()
            _, kwargs = mock_create_engine.call_args

            assert kwargs["echo"] is True
            assert kwargs["pool_size"] == 30
            assert kwargs["max_overflow"] == 50
            assert kwargs["pool_timeout"] == 45
            assert kwargs["pool_pre_ping"] is True

    def test_engine_configuration_postgresql_production(self):
        """Test PostgreSQL production configuration."""
        os.environ.clear()
        os.environ[
            "DATABASE_URL"
        ] = "postgresql://user:secure_pass@localhost/db"
        os.environ["PRODUCTION"] = "true"

        with patch('sqlalchemy.create_engine') as mock_create_engine:
            import importlib

            import database.session

            importlib.reload(database.session)

            _, kwargs = mock_create_engine.call_args

            assert kwargs["pool_recycle"] == 3600
            assert "connect_args" in kwargs
            assert kwargs["connect_args"]["connect_timeout"] == 10
            assert (
                kwargs["connect_args"]["application_name"]
                == "freeagentics_api"
            )

    def test_engine_configuration_sqlite(self):
        """Test SQLite engine configuration."""
        os.environ.clear()
        os.environ["DATABASE_URL"] = "sqlite:///test.db"

        with patch('sqlalchemy.create_engine') as mock_create_engine:
            import importlib

            import database.session

            importlib.reload(database.session)

            _, kwargs = mock_create_engine.call_args

            assert "connect_args" in kwargs
            assert kwargs["connect_args"]["check_same_thread"] is False

    def test_get_db_session_lifecycle(self):
        """Test get_db session lifecycle."""
        os.environ.clear()
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"

        import importlib

        import database.session

        importlib.reload(database.session)

        # Mock SessionLocal
        mock_session = Mock()
        database.session.SessionLocal = Mock(return_value=mock_session)

        # Test session lifecycle
        gen = database.session.get_db()
        session = next(gen)

        assert session == mock_session

        # Complete the generator
        try:
            next(gen)
        except StopIteration:
            pass

        # Session should be closed
        mock_session.close.assert_called_once()

    def test_get_db_session_exception_handling(self):
        """Test get_db closes session on exception."""
        os.environ.clear()
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"

        import importlib

        import database.session

        importlib.reload(database.session)

        mock_session = Mock()
        database.session.SessionLocal = Mock(return_value=mock_session)

        gen = database.session.get_db()
        session = next(gen)

        # Simulate exception
        with pytest.raises(RuntimeError):
            gen.throw(RuntimeError("Test error"))

        # Session should still be closed
        mock_session.close.assert_called_once()

    def test_init_db(self):
        """Test database initialization."""
        os.environ.clear()
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"

        import importlib

        import database.session

        importlib.reload(database.session)

        # Mock Base.metadata
        with patch('database.base.Base.metadata') as mock_metadata:
            database.session.init_db()
            mock_metadata.create_all.assert_called_once_with(
                bind=database.session.engine
            )

    def test_drop_all_tables_development(self):
        """Test dropping tables in development mode."""
        os.environ.clear()
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        os.environ["DEVELOPMENT_MODE"] = "true"

        import importlib

        import database.session

        importlib.reload(database.session)

        with patch('database.base.Base.metadata') as mock_metadata:
            database.session.drop_all_tables()
            mock_metadata.drop_all.assert_called_once_with(
                bind=database.session.engine
            )

    def test_drop_all_tables_production_protection(self):
        """Test protection against dropping tables in production."""
        os.environ.clear()
        os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost/db"
        os.environ["DEVELOPMENT_MODE"] = "false"

        import importlib

        import database.session

        importlib.reload(database.session)

        with pytest.raises(RuntimeError) as exc_info:
            database.session.drop_all_tables()

        assert "Cannot drop tables in production mode" in str(exc_info.value)

    def test_check_database_health_success(self):
        """Test successful database health check."""
        os.environ.clear()
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"

        import importlib

        import database.session

        importlib.reload(database.session)

        # Should return True for working database
        result = database.session.check_database_health()
        assert result is True

    def test_check_database_health_failure(self):
        """Test failed database health check."""
        os.environ.clear()
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"

        import importlib

        import database.session

        importlib.reload(database.session)

        # Mock engine to raise exception
        with patch.object(database.session.engine, 'connect') as mock_connect:
            mock_connect.side_effect = OperationalError(
                "Connection failed", None, None
            )

            result = database.session.check_database_health()
            assert result is False

    def test_postgres_url_format(self):
        """Test handling of postgres:// URL format."""
        os.environ.clear()
        os.environ["DATABASE_URL"] = "postgres://user:pass@localhost/db"

        with patch('sqlalchemy.create_engine') as mock_create_engine:
            import importlib

            import database.session

            importlib.reload(database.session)

            # Should apply PostgreSQL configuration
            _, kwargs = mock_create_engine.call_args
            assert "pool_size" in kwargs
            assert "pool_pre_ping" in kwargs
