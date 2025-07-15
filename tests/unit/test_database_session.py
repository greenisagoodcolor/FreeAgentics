"""Unit tests for database session functionality."""

from unittest.mock import Mock, patch

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from database.session import check_database_health, get_db


class TestGetDb:
    """Test suite for get_db function."""

    @patch("database.session.SessionLocal")
    @patch("database.session.db_state")
    @patch("database.session.db_circuit_breaker")
    def test_get_db_success(self, mock_circuit_breaker, mock_db_state, mock_session_local):
        """Test successful database session creation."""
        # Setup mocks
        mock_session = Mock(spec=Session)
        mock_session_local.return_value = mock_session
        mock_db_state.is_available = True
        mock_circuit_breaker.is_request_allowed.return_value = True
        mock_circuit_breaker.call.return_value = mock_session

        # Test the function
        db_gen = get_db()
        db = next(db_gen)

        assert db == mock_session
        # Circuit breaker should be called to get the session
        mock_circuit_breaker.call.assert_called()

    @patch("database.session.SessionLocal", None)
    def test_get_db_no_session_factory(self):
        """Test error when SessionLocal is not initialized."""
        with pytest.raises(RuntimeError, match="Database session factory not initialized"):
            next(get_db())

    @patch("database.session.SessionLocal")
    @patch("database.session.db_state")
    @patch("database.session.db_circuit_breaker")
    def test_get_db_unavailable_database(
        self, mock_circuit_breaker, mock_db_state, mock_session_local
    ):
        """Test error when database is unavailable."""
        mock_db_state.is_available = False
        mock_db_state.should_retry.return_value = False
        mock_db_state.last_error = "Connection failed"

        with pytest.raises(RuntimeError, match="Database is not available"):
            next(get_db())

    @patch("database.session.SessionLocal")
    @patch("database.session.db_state")
    @patch("database.session.db_circuit_breaker")
    def test_get_db_circuit_breaker_open(
        self, mock_circuit_breaker, mock_db_state, mock_session_local
    ):
        """Test error when circuit breaker is open."""
        mock_db_state.is_available = True
        mock_circuit_breaker.is_request_allowed.return_value = False

        with pytest.raises(Exception):  # CircuitBreakerOpenError
            next(get_db())

    @patch("database.session.SessionLocal")
    @patch("database.session.db_state")
    @patch("database.session.db_circuit_breaker")
    def test_get_db_connection_failure(
        self, mock_circuit_breaker, mock_db_state, mock_session_local
    ):
        """Test handling of database connection failures."""
        mock_session = Mock(spec=Session)
        mock_session_local.return_value = mock_session
        mock_db_state.is_available = True
        mock_circuit_breaker.is_request_allowed.return_value = True

        # The circuit breaker should call the internal function and let it fail
        def circuit_breaker_call(func):
            return func()

        mock_circuit_breaker.call.side_effect = circuit_breaker_call
        # Make the session execute fail to simulate connection failure
        mock_session.execute.side_effect = OperationalError("Connection failed", None, None)

        with pytest.raises(RuntimeError, match="Database connection failed"):
            next(get_db())

    @patch("database.session.SessionLocal")
    @patch("database.session.db_state")
    @patch("database.session.db_circuit_breaker", None)
    def test_get_db_no_circuit_breaker(self, mock_db_state, mock_session_local):
        """Test database session creation without circuit breaker."""
        mock_session = Mock(spec=Session)
        mock_session_local.return_value = mock_session
        mock_db_state.is_available = True

        db_gen = get_db()
        db = next(db_gen)

        assert db == mock_session
        mock_session.execute.assert_called()


class TestCheckDatabaseHealth:
    """Test suite for check_database_health function."""

    @patch("database.session.engine")
    @patch("database.session.db_state")
    @patch("database.session.db_circuit_breaker")
    def test_check_database_health_success(self, mock_circuit_breaker, mock_db_state, mock_engine):
        """Test successful database health check."""
        # Setup mocks
        mock_db_state.is_available = True
        mock_db_state.last_error = None
        mock_db_state.error_count = 0

        mock_circuit_breaker.get_metrics.return_value = {
            "state": "closed",
            "failure_count": 0,
            "success_count": 10,
            "failure_threshold": 5,
            "success_threshold": 3,
            "recovery_timeout": 60,
            "last_failure_time": None,
            "last_success_time": "2023-01-01T12:00:00",
        }

        mock_connection = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection

        def health_check_success():
            return {
                "status": "healthy",
                "details": "Database connection successful",
            }

        mock_circuit_breaker.call.return_value = health_check_success()

        # Test the function
        health_info = check_database_health()

        assert health_info["available"] is True
        assert health_info["last_error"] is None
        assert health_info["error_count"] == 0
        assert "circuit_breaker" in health_info
        assert health_info["status"] == "healthy"

    @patch("database.session.engine", None)
    @patch("database.session.db_state")
    @patch("database.session.db_circuit_breaker")
    def test_check_database_health_no_engine(self, mock_circuit_breaker, mock_db_state):
        """Test health check when engine is not initialized."""
        mock_db_state.is_available = False
        mock_db_state.last_error = "No engine"
        mock_db_state.error_count = 1

        health_info = check_database_health()

        assert health_info["available"] is False
        assert health_info["status"] == "unavailable"
        assert "Database engine not initialized" in health_info["details"]

    @patch("database.session.engine")
    @patch("database.session.db_state")
    @patch("database.session.db_circuit_breaker")
    def test_check_database_health_connection_failure(
        self, mock_circuit_breaker, mock_db_state, mock_engine
    ):
        """Test health check when database connection fails."""
        mock_db_state.is_available = False
        mock_db_state.last_error = "Connection failed"
        mock_db_state.error_count = 5

        mock_circuit_breaker.get_metrics.return_value = {
            "state": "open",
            "failure_count": 5,
            "success_count": 0,
            "failure_threshold": 5,
            "success_threshold": 3,
            "recovery_timeout": 60,
            "last_failure_time": "2023-01-01T12:05:00",
            "last_success_time": None,
        }

        mock_engine.connect.side_effect = OperationalError("Connection failed", None, None)

        health_info = check_database_health()

        assert health_info["available"] is False
        assert health_info["error_count"] == 5
        assert "circuit_breaker" in health_info

    @patch("database.session.engine")
    @patch("database.session.db_state")
    @patch("database.session.db_circuit_breaker")
    def test_check_database_health_circuit_breaker_open(
        self, mock_circuit_breaker, mock_db_state, mock_engine
    ):
        """Test health check when circuit breaker is open."""
        mock_db_state.is_available = False
        mock_db_state.last_error = None
        mock_db_state.error_count = 0

        # Mock circuit breaker exception handling
        class MockCircuitBreakerError(Exception):
            pass

        # Create an exception that contains "CircuitBreakerOpenError" in the type name
        class CircuitBreakerOpenError(MockCircuitBreakerError):
            pass

        mock_circuit_breaker.call.side_effect = CircuitBreakerOpenError("Circuit breaker is open")
        mock_circuit_breaker.get_metrics.return_value = {
            "state": "open",
            "failure_count": 5,
            "success_count": 0,
            "failure_threshold": 5,
            "success_threshold": 3,
            "recovery_timeout": 60,
            "last_failure_time": "2023-01-01T12:05:00",
            "last_success_time": None,
        }

        health_info = check_database_health()

        assert health_info["status"] == "circuit_open"
        assert "circuit breaker is open" in health_info["details"]

    @patch("database.session.engine")
    @patch("database.session.db_state")
    @patch("database.session.db_circuit_breaker", None)
    def test_check_database_health_no_circuit_breaker(self, mock_db_state, mock_engine):
        """Test health check without circuit breaker."""
        mock_db_state.is_available = True
        mock_db_state.last_error = None
        mock_db_state.error_count = 0

        mock_connection = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.pool = Mock()
        mock_engine.pool.size.return_value = 5
        mock_engine.pool.checkedout.return_value = 2

        health_info = check_database_health()

        assert health_info["available"] is True
        assert "circuit_breaker" not in health_info
        assert health_info["status"] == "healthy"

    @patch("database.session.engine")
    @patch("database.session.db_state")
    @patch("database.session.db_circuit_breaker")
    def test_check_database_health_pool_info_error(
        self, mock_circuit_breaker, mock_db_state, mock_engine
    ):
        """Test health check when pool info gathering fails."""
        mock_db_state.is_available = True
        mock_db_state.last_error = None
        mock_db_state.error_count = 0

        # Mock the internal health check to simulate pool error handling
        def health_check_mock():
            # Simulate a health check that succeeds but can't get pool info
            mock_engine.pool = Mock()
            mock_engine.pool.size.side_effect = AttributeError("No pool size")
            return {
                "status": "healthy",
                "details": "Database connection successful",
            }

        mock_circuit_breaker.call.return_value = health_check_mock()
        mock_circuit_breaker.get_metrics.return_value = {
            "state": "closed",
            "failure_count": 0,
            "success_count": 10,
            "failure_threshold": 5,
            "success_threshold": 3,
            "recovery_timeout": 60,
            "last_failure_time": None,
            "last_success_time": "2023-01-01T12:00:00",
        }

        health_info = check_database_health()

        assert health_info["status"] == "healthy"
        assert "details" in health_info
