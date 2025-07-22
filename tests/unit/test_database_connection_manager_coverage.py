"""Comprehensive tests for database.connection_manager module to achieve high coverage."""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLTimeoutError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from database.connection_manager import (
    DatabaseConnectionManager,
    ExponentialBackoffRetry,
)


class TestExponentialBackoffRetry:
    """Test ExponentialBackoffRetry class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        retry = ExponentialBackoffRetry()
        assert retry.max_retries == 3
        assert retry.base_delay == 0.1
        assert retry.max_delay == 1.0
        assert retry.backoff_factor == 2.0

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        retry = ExponentialBackoffRetry(
            max_retries=5,
            base_delay=0.5,
            max_delay=10.0,
            backoff_factor=3.0,
        )
        assert retry.max_retries == 5
        assert retry.base_delay == 0.5
        assert retry.max_delay == 10.0
        assert retry.backoff_factor == 3.0

    def test_calculate_delay(self):
        """Test delay calculation with exponential backoff."""
        retry = ExponentialBackoffRetry(base_delay=0.1, max_delay=2.0, backoff_factor=2.0)

        # Test exponential growth
        assert retry.calculate_delay(0) == 0.1  # 0.1 * 2^0 = 0.1
        assert retry.calculate_delay(1) == 0.2  # 0.1 * 2^1 = 0.2
        assert retry.calculate_delay(2) == 0.4  # 0.1 * 2^2 = 0.4
        assert retry.calculate_delay(3) == 0.8  # 0.1 * 2^3 = 0.8
        assert retry.calculate_delay(4) == 1.6  # 0.1 * 2^4 = 1.6

        # Test max_delay cap
        assert retry.calculate_delay(5) == 2.0  # Would be 3.2, capped at 2.0
        assert retry.calculate_delay(10) == 2.0  # Would be much higher, capped at 2.0

    def test_execute_with_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        retry = ExponentialBackoffRetry()

        # Mock function that succeeds
        mock_func = Mock(return_value="success")

        result = retry.execute_with_retry(mock_func, "arg1", key="value")

        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_with("arg1", key="value")

    def test_execute_with_retry_success_after_failures(self):
        """Test successful execution after some failures."""
        retry = ExponentialBackoffRetry(max_retries=3)

        # Mock function that fails twice then succeeds
        mock_func = Mock(
            side_effect=[
                OperationalError("Connection failed", None, None),
                SQLTimeoutError("Timeout", None, None),
                "success",
            ]
        )

        with patch.object(retry, "calculate_delay", return_value=0.1):
            with patch("numpy.random.bytes") as mock_bytes:
                mock_bytes.return_value = b"test"

                result = retry.execute_with_retry(mock_func)

                assert result == "success"
                assert mock_func.call_count == 3
                assert mock_bytes.call_count == 2  # Called for each retry

    def test_execute_with_retry_all_attempts_fail(self):
        """Test when all retry attempts fail."""
        retry = ExponentialBackoffRetry(max_retries=3)

        # Mock function that always fails
        exception = OperationalError("Connection failed", None, None)
        mock_func = Mock(side_effect=exception)

        with patch.object(retry, "calculate_delay", return_value=0.1):
            with patch("numpy.random.bytes") as mock_bytes:
                mock_bytes.return_value = b"test"

                with pytest.raises(OperationalError) as exc_info:
                    retry.execute_with_retry(mock_func)

                assert exc_info.value is exception
                assert mock_func.call_count == 3
                assert mock_bytes.call_count == 2  # Not called on last attempt

    def test_execute_with_retry_connection_error(self):
        """Test retry with ConnectionError."""
        retry = ExponentialBackoffRetry(max_retries=2)

        # Mock function that raises ConnectionError
        mock_func = Mock(side_effect=ConnectionError("Network unreachable"))

        with patch("numpy.random.bytes") as mock_bytes:
            mock_bytes.return_value = b"test"

            with pytest.raises(ConnectionError):
                retry.execute_with_retry(mock_func)

            assert mock_func.call_count == 2

    def test_execute_with_retry_non_retryable_exception(self):
        """Test that non-retryable exceptions are raised immediately."""
        retry = ExponentialBackoffRetry()

        # Mock function that raises non-retryable exception
        mock_func = Mock(side_effect=ValueError("Invalid parameter"))

        with pytest.raises(ValueError) as exc_info:
            retry.execute_with_retry(mock_func)

        assert str(exc_info.value) == "Invalid parameter"
        assert mock_func.call_count == 1  # No retries for non-retryable exceptions

    def test_execute_with_retry_no_exception_runtime_error(self):
        """Test RuntimeError when retry logic fails without exception."""
        retry = ExponentialBackoffRetry(max_retries=1)

        # Mock function that returns None (simulating unexpected behavior)
        def mock_func():
            retry.max_retries = 0  # Manipulate to break the loop
            return None

        # This is a edge case that shouldn't happen in practice
        # but we test it for coverage
        with patch.object(retry, "max_retries", 0):
            with pytest.raises(RuntimeError) as exc_info:
                retry.execute_with_retry(lambda: None)

            assert "Retry logic failed without capturing exception" in str(exc_info.value)

    def test_execute_with_retry_logging(self, caplog):
        """Test logging during retry attempts."""
        retry = ExponentialBackoffRetry(max_retries=3)

        mock_func = Mock(
            side_effect=[
                OperationalError("Connection 1", None, None),
                OperationalError("Connection 2", None, None),
                "success",
            ]
        )

        with patch("numpy.random.bytes", return_value=b"test"):
            with caplog.at_level(logging.WARNING):
                result = retry.execute_with_retry(mock_func)

                assert result == "success"
                assert len(caplog.records) == 2
                assert "attempt 1 failed" in caplog.records[0].message
                assert "attempt 2 failed" in caplog.records[1].message

    def test_execute_with_retry_logging_all_fail(self, caplog):
        """Test logging when all attempts fail."""
        retry = ExponentialBackoffRetry(max_retries=2)

        mock_func = Mock(side_effect=OperationalError("Failed", None, None))

        with patch("numpy.random.bytes", return_value=b"test"):
            with caplog.at_level(logging.ERROR):
                with pytest.raises(OperationalError):
                    retry.execute_with_retry(mock_func)

                error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
                assert len(error_logs) == 1
                assert "All 2 connection attempts failed" in error_logs[0].message


class TestDatabaseConnectionManager:
    """Test DatabaseConnectionManager class."""

    def test_init(self):
        """Test initialization."""
        manager = DatabaseConnectionManager("postgresql://localhost/test")

        assert manager.database_url == "postgresql://localhost/test"
        assert isinstance(manager.retry_handler, ExponentialBackoffRetry)
        assert manager._engine is None
        assert manager._async_engine is None
        assert manager._session_factory is None
        assert manager._async_session_factory is None

    def test_get_connection_with_retry_success(self):
        """Test successful connection with retry."""
        manager = DatabaseConnectionManager("sqlite:///:memory:")

        with patch("database.connection_manager.create_engine") as mock_create_engine:
            mock_engine = Mock()
            mock_conn = Mock()
            mock_result = Mock()
            mock_result.scalar.return_value = 1

            mock_conn.execute.return_value = mock_result
            mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
            mock_engine.connect.return_value.__exit__ = Mock(return_value=None)

            mock_create_engine.return_value = mock_engine

            result = manager.get_connection_with_retry()

            assert result == mock_engine
            mock_create_engine.assert_called_with("sqlite:///:memory:")
            mock_conn.execute.assert_called_once()

    def test_get_connection_with_retry_custom_retries(self):
        """Test connection with custom retry count."""
        manager = DatabaseConnectionManager("sqlite:///:memory:")

        with patch("database.connection_manager.create_engine") as mock_create_engine:
            # Create a series of mock engines - first few fail, last succeeds
            failed_engine1 = Mock()
            failed_conn1 = Mock()
            failed_conn1.execute.side_effect = OperationalError("Failed", None, None)
            failed_engine1.connect.return_value.__enter__ = Mock(return_value=failed_conn1)
            failed_engine1.connect.return_value.__exit__ = Mock(return_value=None)

            failed_engine2 = Mock()
            failed_conn2 = Mock()
            failed_conn2.execute.side_effect = OperationalError("Failed", None, None)
            failed_engine2.connect.return_value.__enter__ = Mock(return_value=failed_conn2)
            failed_engine2.connect.return_value.__exit__ = Mock(return_value=None)

            success_engine = Mock()
            success_conn = Mock()
            success_conn.execute.return_value = Mock()
            success_engine.connect.return_value.__enter__ = Mock(return_value=success_conn)
            success_engine.connect.return_value.__exit__ = Mock(return_value=None)

            mock_create_engine.side_effect = [
                failed_engine1,
                failed_engine2,
                success_engine,
            ]

            with patch("numpy.random.bytes", return_value=b"test"):
                result = manager.get_connection_with_retry(max_retries=3)

                assert result == success_engine
                assert mock_create_engine.call_count == 3

    def test_get_db_session(self):
        """Test getting database session."""
        manager = DatabaseConnectionManager("sqlite:///:memory:")

        # Setup mocks
        mock_session = Mock(spec=Session)
        mock_session.execute = Mock()
        mock_factory = Mock(return_value=mock_session)

        with patch.object(manager, "get_session_factory", return_value=mock_factory):
            session = manager.get_db_session()

            assert session == mock_session
            mock_factory.assert_called_once()
            mock_session.execute.assert_called_once()

    def test_get_session_factory(self):
        """Test getting session factory."""
        manager = DatabaseConnectionManager("sqlite:///:memory:")

        mock_engine = Mock()
        manager._engine = mock_engine

        result = manager.get_session_factory()

        assert result is not None
        assert manager._session_factory is not None
        assert manager._session_factory == result

    def test_create_engine_with_pool_config(self):
        """Test creating engine with pool configuration."""
        # Use PostgreSQL URL to allow pool config
        manager = DatabaseConnectionManager("postgresql://user:pass@localhost/test")

        with patch("database.connection_manager.create_engine") as mock_create_engine:
            mock_engine = Mock()
            mock_conn = Mock()
            mock_conn.execute = Mock()
            mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
            mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
            mock_create_engine.return_value = mock_engine

            result = manager.create_engine_with_pool_config(pool_size=20)

            assert result == mock_engine
            assert manager._engine == mock_engine

            # Check pool config was passed
            mock_create_engine.assert_called_once()
            _, kwargs = mock_create_engine.call_args
            assert kwargs["pool_size"] == 20
            assert kwargs["pool_pre_ping"] is True

    async def test_get_async_db_session(self):
        """Test getting async database session."""
        manager = DatabaseConnectionManager("postgresql+asyncpg://localhost/test")

        # Setup async mocks
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock()
        mock_factory = Mock(return_value=mock_session)
        manager._async_session_factory = mock_factory
        manager._async_engine = Mock()  # Prevent creation

        session = await manager.get_async_db_session()

        assert session == mock_session
        mock_factory.assert_called_once()
        mock_session.execute.assert_called_once()

    def test_create_async_engine_pool(self):
        """Test creating async engine pool."""
        manager = DatabaseConnectionManager("postgresql://localhost/test")

        with patch("database.connection_manager.create_async_engine") as mock_create:
            mock_engine = Mock()
            mock_create.return_value = mock_engine

            result = manager.create_async_engine_pool()

            assert result == mock_engine
            assert manager._async_engine == mock_engine

            # Check URL was converted to asyncpg
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert args[0] == "postgresql+asyncpg://localhost/test"
            assert kwargs["pool_size"] == 10
            assert kwargs["pool_pre_ping"] is True
