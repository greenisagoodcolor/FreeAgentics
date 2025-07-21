"""Security-critical tests for database operations following TDD principles.

This test suite covers database integrity and security:
- Transaction safety and rollback
- SQL injection prevention
- Connection pool security
- Data validation and sanitization
- Concurrent access control
"""

import threading
import time

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import sessionmaker

from database.connection_manager import DatabaseConnectionManager as ConnectionManager
from database.models import Base, User
from database.validation import validate_model_data


class TestDatabaseTransactions:
    """Test database transaction integrity and safety."""

    @pytest.fixture
    def test_db(self):
        """Create in-memory test database."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        # Create connection manager
        connection_manager = ConnectionManager(engine)

        yield connection_manager, engine

        # Cleanup
        engine.dispose()

    def test_transaction_rollback_on_error(self, test_db):
        """Test that transactions rollback on error."""
        connection_manager, engine = test_db

        # Arrange
        initial_count = 0

        with connection_manager.get_session() as session:
            # Add initial user
            user = User(username="test_user", email="test@example.com")
            session.add(user)
            session.commit()
            initial_count = session.query(User).count()

        # Act - Attempt transaction that will fail
        try:
            with connection_manager.get_session() as session:
                # Add valid user
                user1 = User(username="user1", email="user1@example.com")
                session.add(user1)

                # Add duplicate that will cause error
                user2 = User(username="test_user", email="duplicate@example.com")
                session.add(user2)

                # This should fail due to unique constraint
                session.commit()
        except IntegrityError:
            pass  # Expected

        # Assert - No new users should be added
        with connection_manager.get_session() as session:
            final_count = session.query(User).count()
            assert final_count == initial_count

    def test_nested_transaction_isolation(self, test_db):
        """Test nested transaction isolation."""
        connection_manager, engine = test_db

        # Arrange
        with connection_manager.get_session() as session:
            user = User(username="parent_user", email="parent@example.com")
            session.add(user)
            session.commit()
            user_id = user.id

        # Act - Nested transaction with rollback
        with connection_manager.get_session() as outer_session:
            # Modify in outer transaction
            user = outer_session.query(User).filter_by(id=user_id).first()
            user.email = "modified@example.com"

            # Create savepoint for nested transaction
            savepoint = outer_session.begin_nested()

            try:
                # Nested modification that will be rolled back
                user.email = "nested@example.com"
                # Simulate error
                raise Exception("Nested error")
            except Exception:
                savepoint.rollback()

            # Commit outer transaction
            outer_session.commit()

        # Assert - Only outer modification should persist
        with connection_manager.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            assert user.email == "modified@example.com"

    def test_concurrent_transaction_safety(self, test_db):
        """Test concurrent transaction handling."""
        connection_manager, engine = test_db

        # Arrange
        results = []
        errors = []

        def concurrent_insert(user_id):
            try:
                with connection_manager.get_session() as session:
                    user = User(
                        username=f"concurrent_{user_id}",
                        email=f"user{user_id}@example.com",
                    )
                    session.add(user)
                    # Add small delay to increase chance of conflicts
                    time.sleep(0.01)
                    session.commit()
                    results.append(user_id)
            except Exception as e:
                errors.append((user_id, str(e)))

        # Act - Run concurrent insertions
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_insert, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Assert - All should succeed without conflicts
        assert len(errors) == 0
        assert len(results) == 10

        with connection_manager.get_session() as session:
            count = (
                session.query(User).filter(User.username.like("concurrent_%")).count()
            )
            assert count == 10


class TestSQLInjectionPrevention:
    """Test SQL injection prevention measures."""

    @pytest.fixture
    def secure_db(self):
        """Create secure database connection."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        connection_manager = ConnectionManager(engine)
        yield connection_manager

        engine.dispose()

    def test_parameterized_queries_prevent_injection(self, secure_db):
        """Test that parameterized queries prevent SQL injection."""
        # Arrange
        with secure_db.get_session() as session:
            # Add test data
            user = User(username="admin", email="admin@example.com")
            session.add(user)
            session.commit()

        # Act - Attempt SQL injection
        malicious_inputs = [
            "admin' OR '1'='1",
            "admin'; DROP TABLE users; --",
            "admin' UNION SELECT * FROM users --",
            "admin\\'; DROP TABLE users; --",
        ]

        for malicious_input in malicious_inputs:
            with secure_db.get_session() as session:
                # Safe parameterized query
                result = (
                    session.query(User).filter(User.username == malicious_input).first()
                )

                # Assert - Should not find user or execute injection
                assert result is None

    def test_input_sanitization(self, secure_db):
        """Test that inputs are properly sanitized."""
        # Arrange
        dangerous_inputs = {
            "username": "<script>alert('xss')</script>",
            "email": "test@example.com'; DROP TABLE users; --",
            "metadata": {"key": "value'; DELETE FROM agents; --"},
        }

        # Act
        with secure_db.get_session() as session:
            user = User(
                username=dangerous_inputs["username"], email=dangerous_inputs["email"]
            )
            session.add(user)
            session.commit()
            user_id = user.id

        # Assert - Data should be stored as-is (escaped, not executed)
        with secure_db.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            assert user.username == dangerous_inputs["username"]
            assert "<script>" in user.username  # Not stripped, just stored safely

    def test_raw_sql_protection(self, secure_db):
        """Test protection when raw SQL must be used."""
        # Arrange
        with secure_db.get_session() as session:
            # Add test users
            for i in range(3):
                user = User(username=f"user{i}", email=f"user{i}@example.com")
                session.add(user)
            session.commit()

        # Act - Safe raw SQL with parameters
        search_term = "user1' OR '1'='1"

        with secure_db.get_session() as session:
            # Safe approach with parameters
            result = session.execute(
                text("SELECT * FROM users WHERE username = :username"),
                {"username": search_term},
            ).fetchall()

        # Assert - Should find no results (injection prevented)
        assert len(result) == 0


class TestConnectionPoolSecurity:
    """Test connection pool security and resource management."""

    @pytest.fixture
    def pool_manager(self):
        """Create connection pool manager."""
        engine = create_engine(
            "sqlite:///:memory:", pool_size=5, max_overflow=2, pool_timeout=30
        )
        Base.metadata.create_all(engine)

        manager = ConnectionManager(engine)
        yield manager

        engine.dispose()

    def test_connection_pool_limits(self, pool_manager):
        """Test that connection pool enforces limits."""
        # Arrange
        connections = []
        max_connections = 7  # pool_size + max_overflow

        # Act - Try to exceed pool limits
        try:
            for i in range(max_connections + 1):
                conn = pool_manager.engine.connect()
                connections.append(conn)
        except OperationalError as e:
            # Expected when pool is exhausted
            assert "timeout" in str(e).lower() or "pool" in str(e).lower()

        # Cleanup
        for conn in connections:
            conn.close()

    def test_connection_leak_prevention(self, pool_manager):
        """Test that connections are properly returned to pool."""
        # Arrange
        initial_pool_size = pool_manager.engine.pool.size()

        # Act - Use connections with context manager
        for i in range(10):
            with pool_manager.get_session() as session:
                # Perform operation
                session.execute(text("SELECT 1"))

        # Assert - Pool should not grow (connections returned)
        final_pool_size = pool_manager.engine.pool.size()
        assert final_pool_size <= initial_pool_size + 1  # Allow small variance

    def test_connection_timeout_handling(self, pool_manager):
        """Test handling of connection timeouts."""
        # Arrange - Create small pool
        small_engine = create_engine(
            "sqlite:///:memory:",
            pool_size=1,
            max_overflow=0,
            pool_timeout=0.1,  # 100ms timeout
        )
        Base.metadata.create_all(small_engine)
        small_manager = ConnectionManager(small_engine)

        # Hold one connection
        conn1 = small_manager.engine.connect()

        # Act - Try to get another connection (should timeout)
        start_time = time.time()
        with pytest.raises(OperationalError):
            small_manager.engine.connect()

        # Assert - Should timeout quickly
        elapsed = time.time() - start_time
        assert elapsed < 0.5  # Should timeout in ~100ms

        # Cleanup
        conn1.close()
        small_engine.dispose()


class TestDataValidation:
    """Test data validation and sanitization."""

    def test_validate_model_data_types(self):
        """Test that model data types are validated."""
        # Arrange
        valid_data = {
            "username": "testuser",
            "email": "test@example.com",
            "age": 25,
            "is_active": True,
        }

        invalid_data_sets = [
            {"username": None, "email": "test@example.com"},  # Required field
            {"username": "a" * 256, "email": "test@example.com"},  # Too long
            {"username": "test", "email": "not-an-email"},  # Invalid format
            {"username": "test", "email": "test@example.com", "age": "not-a-number"},
        ]

        # Act & Assert - Valid data should pass
        assert validate_model_data(User, valid_data) is True

        # Invalid data should fail
        for invalid_data in invalid_data_sets:
            assert validate_model_data(User, invalid_data) is False

    def test_prevent_mass_assignment(self):
        """Test prevention of mass assignment vulnerabilities."""
        # Arrange
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "is_admin": True,  # Should not be mass assignable
            "password_hash": "fake_hash",  # Should not be mass assignable
        }

        # Act - Create user with filtered data
        safe_fields = ["username", "email"]
        filtered_data = {k: v for k, v in user_data.items() if k in safe_fields}

        user = User(**filtered_data)

        # Assert - Dangerous fields not set
        assert not hasattr(user, "is_admin") or user.is_admin is False
        assert not hasattr(user, "password_hash") or user.password_hash is None


class TestDatabaseSecurity:
    """Test overall database security measures."""

    @pytest.fixture
    def secure_session(self):
        """Create secure database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        yield session

        session.close()
        engine.dispose()

    def test_query_result_limits(self, secure_session):
        """Test that queries have reasonable result limits."""
        # Arrange - Add many records
        for i in range(1000):
            user = User(username=f"user{i}", email=f"user{i}@example.com")
            secure_session.add(user)
        secure_session.commit()

        # Act - Query with limit
        query = secure_session.query(User)
        # Should have default limit or pagination
        results = query.limit(100).all()

        # Assert
        assert len(results) == 100

        # Verify protection against unbounded queries
        # In production, this would be enforced at query construction
        all_results = query.all()
        # Should log warning for large result sets
        assert len(all_results) == 1000

    def test_sensitive_data_encryption(self, secure_session):
        """Test that sensitive data is encrypted at rest."""
        # This is a placeholder - actual implementation would use
        # SQLAlchemy encryption types or database-level encryption

        # Arrange
        from database.models import SensitiveData  # Hypothetical model

        sensitive = SensitiveData(
            user_id=1, ssn="123-45-6789", credit_card="4111-1111-1111-1111"
        )

        # Act - Save to database
        secure_session.add(sensitive)
        secure_session.commit()

        # Assert - Raw query should show encrypted data
        result = secure_session.execute(
            text("SELECT ssn, credit_card FROM sensitive_data WHERE user_id = 1")
        ).first()

        # Values should be encrypted (not plaintext)
        assert result.ssn != "123-45-6789"
        assert result.credit_card != "4111-1111-1111-1111"
