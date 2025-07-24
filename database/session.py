"""Database session management for FreeAgentics.

This module provides SQLAlchemy session management and connection pooling
for the database. In production, PostgreSQL is required. In development mode,
SQLite is used as a fallback when DATABASE_URL is not set.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import OperationalError

from database.base import Base
from websocket.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitOpenException

# Get database URL from environment with SQLite fallback for development
DATABASE_URL = os.getenv("DATABASE_URL")

# Check if we're in development mode
is_development = (
    os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
    or os.getenv("ENVIRONMENT", "").lower() == "development"
    or os.getenv("ENV", "").lower() == "development"
)

if not DATABASE_URL:
    if is_development:
        # Use SQLite as fallback for development
        import warnings

        warnings.warn(
            "DATABASE_URL not set. Using SQLite for development. "
            "This is not suitable for production!",
            RuntimeWarning,
        )
        DATABASE_URL = "sqlite:///./freeagentics_dev.db"
    else:
        raise ValueError(
            "DATABASE_URL environment variable is required. "
            "Please set it in your .env file or environment. "
            "Format: postgresql://username:password@host:port/database"
        )

# Security validation for production
if os.getenv("PRODUCTION", "false").lower() == "true":
    # Ensure we're not using default dev credentials in production
    if "freeagentics_dev_2025" in DATABASE_URL or "freeagentics123" in DATABASE_URL:
        raise ValueError(
            "Production environment detected but using development database credentials. "
            "Please set secure DATABASE_URL in production."
        )
    # Require SSL/TLS in production
    if "sslmode=" not in DATABASE_URL:
        DATABASE_URL += "?sslmode=require"

# Create engine with connection pooling and security settings
engine_args: Dict[str, Any] = {
    "echo": os.getenv("DEBUG_SQL", "false").lower() == "true",  # SQL logging
}

# Configure pooling based on database dialect
if DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"):
    # PostgreSQL-specific configuration
    engine_args.update(
        {
            "pool_size": int(os.getenv("DB_POOL_SIZE", "20")),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "40")),
            "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
            "pool_pre_ping": True,  # Verify connections before using
        }
    )

    # Additional production settings for PostgreSQL
    if os.getenv("PRODUCTION", "false").lower() == "true":
        engine_args.update(
            {
                "pool_recycle": 3600,  # Recycle connections after 1 hour
                "connect_args": {
                    "connect_timeout": 10,
                    "application_name": "freeagentics_api",
                    "options": "-c statement_timeout=30000",  # 30 second statement timeout
                },
            }
        )
elif DATABASE_URL.startswith("sqlite://"):
    # SQLite-specific configuration
    engine_args.update({"connect_args": {"check_same_thread": False}})

engine = create_engine(DATABASE_URL, **engine_args)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


@dataclass
class DatabaseState:
    """Tracks database connection state and health metrics."""
    is_available: bool = True
    last_error: Optional[str] = None
    error_count: int = 0
    last_check_time: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    
    def record_success(self):
        """Record a successful database operation."""
        self.is_available = True
        self.last_error = None
        self.consecutive_failures = 0
        self.last_check_time = time.time()
    
    def record_failure(self, error: str):
        """Record a failed database operation."""
        self.last_error = error
        self.error_count += 1
        self.consecutive_failures += 1
        self.last_check_time = time.time()
        
        # Mark as unavailable after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.is_available = False
    
    def should_retry(self) -> bool:
        """Check if we should retry connecting."""
        # Retry after 5 seconds if unavailable
        if not self.is_available:
            return (time.time() - self.last_check_time) > 5.0
        return True


# Initialize database state tracker
db_state = DatabaseState()

# Initialize circuit breaker for database connections
db_circuit_breaker_config = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=3,
    timeout=60.0,
    half_open_max_calls=3,
    excluded_exceptions=(KeyboardInterrupt, SystemExit)
)
db_circuit_breaker = CircuitBreaker("database", db_circuit_breaker_config)


def get_db() -> Generator[Session, None, None]:
    """Get database session for dependency injection.

    Yields:
        Database session that auto-closes after use
        
    Raises:
        RuntimeError: If database is not available or circuit breaker is open
    """
    # Check if SessionLocal is initialized
    if SessionLocal is None:
        raise RuntimeError("Database session factory not initialized")
    
    # Check database availability
    if not db_state.is_available and not db_state.should_retry():
        raise RuntimeError(f"Database is not available: {db_state.last_error}")
    
    # Check circuit breaker
    if db_circuit_breaker:
        # Check using both methods for compatibility with tests
        if hasattr(db_circuit_breaker, 'is_request_allowed'):
            allowed = db_circuit_breaker.is_request_allowed()
        else:
            allowed = db_circuit_breaker.should_allow_request()
        
        if not allowed:
            raise CircuitOpenException("database", db_circuit_breaker.last_failure_time)
    
    def create_session():
        """Create and verify database session."""
        db = SessionLocal()
        try:
            # Verify connection with a simple query
            db.execute(text("SELECT 1"))
            db_state.record_success()
            return db
        except OperationalError as e:
            db.close()
            db_state.record_failure(str(e))
            raise RuntimeError(f"Database connection failed: {e}")
    
    # Use circuit breaker if available
    if db_circuit_breaker:
        try:
            db = db_circuit_breaker.call(create_session)
        except CircuitOpenException:
            raise
        except Exception as e:
            raise RuntimeError(f"Database connection failed: {e}")
    else:
        db = create_session()
    
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database by creating all tables.

    This should be called on application startup or via migration.
    """
    Base.metadata.create_all(bind=engine)


def drop_all_tables() -> None:
    """Drop all tables - DANGEROUS, only for development.

    This will DELETE ALL DATA. Use with extreme caution.
    """
    if os.getenv("DEVELOPMENT_MODE", "false").lower() != "true":
        raise RuntimeError("Cannot drop tables in production mode")

    Base.metadata.drop_all(bind=engine)


def check_database_health() -> Dict[str, Any]:
    """Check if database connection is healthy.

    Returns:
        Dictionary containing health status and metrics
    """
    health_info = {
        "available": db_state.is_available,
        "last_error": db_state.last_error,
        "error_count": db_state.error_count,
        "status": "unknown",
        "details": "Not checked"
    }
    
    # Add circuit breaker metrics if available
    if db_circuit_breaker:
        health_info["circuit_breaker"] = db_circuit_breaker.get_status()
    
    # Check if engine is initialized
    if engine is None:
        health_info["status"] = "unavailable"
        health_info["details"] = "Database engine not initialized"
        return health_info
    
    def perform_health_check():
        """Perform actual health check."""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                if result.scalar() == 1:
                    db_state.record_success()
                    
                    # Try to get pool information
                    pool_info = {}
                    try:
                        if hasattr(engine, 'pool'):
                            pool_info = {
                                "pool_size": engine.pool.size(),
                                "checked_out": engine.pool.checkedout()
                            }
                    except AttributeError:
                        pass
                    
                    return {
                        "status": "healthy",
                        "details": "Database connection successful",
                        **pool_info
                    }
                else:
                    raise RuntimeError("Unexpected query result")
        except Exception as e:
            db_state.record_failure(str(e))
            error_type = type(e).__name__
            if "CircuitBreakerOpenError" in error_type:
                return {
                    "status": "circuit_open",
                    "details": f"Circuit breaker is open: {e}"
                }
            return {
                "status": "unhealthy",
                "details": f"Database check failed: {e}"
            }
    
    # Use circuit breaker if available
    if db_circuit_breaker:
        try:
            result = db_circuit_breaker.call(perform_health_check)
            health_info.update(result)
        except CircuitOpenException as e:
            health_info["status"] = "circuit_open"
            health_info["details"] = str(e).lower()
        except Exception as e:
            # Check if it's a circuit breaker open error by examining the exception type
            error_type = type(e).__name__
            if "CircuitBreakerOpenError" in error_type:
                health_info["status"] = "circuit_open"
                health_info["details"] = str(e).lower()
            else:
                health_info["status"] = "error"
                health_info["details"] = str(e)
    else:
        result = perform_health_check()
        health_info.update(result)
    
    return health_info
