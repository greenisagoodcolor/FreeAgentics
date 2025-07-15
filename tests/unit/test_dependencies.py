"""
Test-Driven Development for Dependency Installation and Verification

Following RED-GREEN-REFACTOR cycle:
1. RED: These tests must fail initially to expose missing/incorrect dependencies
2. GREEN: Implement minimal dependency installation to pass tests
3. REFACTOR: Clean up dependency configuration

NO graceful fallbacks - dependencies must be available or tests fail hard.
"""

import pytest
from packaging import version


class TestDependencyVerification:
    """Test suite for verifying all required dependencies are properly installed."""

    def test_fastapi_available_and_correct_version(self):
        """Test FastAPI is available and meets minimum version requirements."""
        try:
            import fastapi

            # Minimum version requirement for production features
            assert version.parse(fastapi.__version__) >= version.parse("0.100.0")
        except ImportError:
            pytest.fail("FastAPI is not installed - required for API endpoints")
        except AttributeError:
            pytest.fail("FastAPI installation is corrupted - no version attribute")

    def test_sqlalchemy_available_and_correct_version(self):
        """Test SQLAlchemy is available and meets minimum version for async support."""
        try:
            import sqlalchemy

            # Minimum version for proper async support
            assert version.parse(sqlalchemy.__version__) >= version.parse("2.0.0")
        except ImportError:
            pytest.fail("SQLAlchemy is not installed - required for database operations")
        except AttributeError:
            pytest.fail("SQLAlchemy installation is corrupted - no version attribute")

    def test_psycopg2_available(self):
        """Test psycopg2 is available for PostgreSQL connectivity."""
        try:
            import psycopg2

            # Verify it can create a connection string
            assert hasattr(psycopg2, "connect")
        except ImportError:
            pytest.fail("psycopg2 is not installed - required for PostgreSQL connectivity")

    def test_asyncpg_available(self):
        """Test asyncpg is available for async PostgreSQL operations."""
        try:
            import asyncpg

            assert hasattr(asyncpg, "connect")
        except ImportError:
            pytest.fail("asyncpg is not installed - required for async database operations")

    def test_passlib_available_with_bcrypt(self):
        """Test passlib is available with bcrypt support for password hashing."""
        try:
            from passlib.context import CryptContext

            # Test bcrypt handler is available
            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            test_hash = pwd_context.hash("test")
            assert pwd_context.verify("test", test_hash)
        except ImportError:
            pytest.fail("passlib is not installed - required for password hashing")
        except Exception as e:
            pytest.fail(f"passlib bcrypt support failed: {e}")

    def test_pyjwt_available_and_functional(self):
        """Test PyJWT is available and can encode/decode tokens."""
        try:
            import jwt

            # Test basic JWT functionality
            payload = {"test": "data"}
            secret = "test-secret"
            token = jwt.encode(payload, secret, algorithm="HS256")
            decoded = jwt.decode(token, secret, algorithms=["HS256"])
            assert decoded["test"] == "data"
        except ImportError:
            pytest.fail("PyJWT is not installed - required for authentication")
        except Exception as e:
            pytest.fail(f"PyJWT functionality test failed: {e}")

    def test_python_jose_cryptography_available(self):
        """Test python-jose with cryptography backend is available."""
        try:
            from jose import jwt as jose_jwt

            # Test that basic JWT functionality works
            assert hasattr(jose_jwt, "encode")
            assert hasattr(jose_jwt, "decode")

            # Test actual cryptography functionality with RS256
            import cryptography.hazmat.primitives.asymmetric.rsa as rsa
            from cryptography.hazmat.primitives import serialization

            # Generate test key pair for RS256
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            public_key = private_key.public_key()

            # Convert to PEM format
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            # Test JWT encode/decode with RS256
            payload = {"test": "data"}
            token = jose_jwt.encode(payload, private_pem, algorithm="RS256")
            decoded = jose_jwt.decode(token, public_pem, algorithms=["RS256"])
            assert decoded["test"] == "data"

        except ImportError:
            pytest.fail("python-jose[cryptography] is not installed - required for JWT RS256")
        except Exception as e:
            pytest.fail(f"python-jose cryptography functionality test failed: {e}")

    def test_uvicorn_available_and_correct_version(self):
        """Test uvicorn is available for ASGI server."""
        try:
            import uvicorn

            # Check version for security and performance fixes
            assert version.parse(uvicorn.__version__) >= version.parse("0.20.0")
        except ImportError:
            pytest.fail("uvicorn is not installed - required for running FastAPI server")
        except AttributeError:
            pytest.fail("uvicorn installation is corrupted - no version attribute")

    def test_pydantic_v2_available(self):
        """Test Pydantic v2 is available for data validation."""
        try:
            import pydantic

            # Ensure we have v2 for performance and features
            assert version.parse(pydantic.__version__) >= version.parse("2.0.0")
            # Test v2 features are available
            from pydantic import BaseModel, Field

            class TestModel(BaseModel):
                name: str = Field(..., min_length=1)

            # This should work in v2
            test_instance = TestModel(name="test")
            assert test_instance.name == "test"
        except ImportError:
            pytest.fail("Pydantic is not installed - required for data validation")
        except Exception as e:
            pytest.fail(f"Pydantic v2 functionality test failed: {e}")

    def test_redis_client_available(self):
        """Test Redis client is available for caching and pub/sub."""
        try:
            import redis

            # Test that Redis client can be instantiated
            client = redis.Redis()
            assert hasattr(client, "ping")
        except ImportError:
            pytest.fail("redis is not installed - required for caching and real-time updates")

    def test_alembic_available_for_migrations(self):
        """Test Alembic is available for database migrations."""
        try:
            import alembic
            import alembic.command
            from alembic.config import Config

            # Test that we can create a config (even with dummy path)
            Config()
            assert hasattr(alembic.command, "upgrade")
            assert hasattr(alembic.command, "downgrade")
        except ImportError as e:
            pytest.fail(f"alembic is not installed - required for database migrations: {e}")
        except Exception as e:
            pytest.fail(f"alembic functionality test failed: {e}")


class TestDependencyCompatibility:
    """Test that dependencies work together without conflicts."""

    def test_sqlalchemy_fastapi_integration(self):
        """Test SQLAlchemy works with FastAPI dependency injection."""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            # This should not raise any compatibility errors
            engine = create_engine("sqlite:///:memory:")
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

            def get_db():
                db = SessionLocal()
                try:
                    yield db
                finally:
                    db.close()

            # Test dependency creation doesn't fail
            assert callable(get_db)
        except Exception as e:
            pytest.fail(f"SQLAlchemy-FastAPI integration failed: {e}")

    def test_pydantic_sqlalchemy_integration(self):
        """Test Pydantic models work with SQLAlchemy."""
        try:
            from pydantic import BaseModel
            from sqlalchemy import Column, Integer, String
            from sqlalchemy.ext.declarative import declarative_base

            Base = declarative_base()

            class User(Base):
                __tablename__ = "users"
                id = Column(Integer, primary_key=True)
                name = Column(String(50))

            class UserResponse(BaseModel):
                id: int
                name: str

                class Config:
                    from_attributes = True

            # Test models are compatible
            assert hasattr(UserResponse, "model_validate")
        except Exception as e:
            pytest.fail(f"Pydantic-SQLAlchemy integration failed: {e}")

    def test_async_dependencies_available(self):
        """Test async support libraries are available and compatible."""
        try:
            import asyncio

            import asyncpg

            # Import the redis library for async operations
            import redis.asyncio as aioredis

            # Test async function definition doesn't fail
            async def test_async():
                return True

            # Test asyncio can run the function
            result = asyncio.run(test_async())
            assert result is True

            # Test that async libraries have expected attributes
            assert hasattr(asyncpg, "connect")
            assert hasattr(aioredis, "Redis")
        except Exception as e:
            pytest.fail(f"Async dependencies test failed: {e}")


class TestCriticalDependencyImports:
    """Test that critical imports work without errors - hard failures only."""

    def test_critical_fastapi_imports(self):
        """Test critical FastAPI imports work."""
        try:
            from fastapi import Depends, FastAPI, HTTPException, status
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import JSONResponse
            from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

            # These must all succeed
            assert all(
                [
                    FastAPI,
                    HTTPException,
                    Depends,
                    status,
                    HTTPBearer,
                    HTTPAuthorizationCredentials,
                    CORSMiddleware,
                    JSONResponse,
                ]
            )
        except ImportError as e:
            pytest.fail(f"Critical FastAPI import failed: {e}")

    def test_critical_database_imports(self):
        """Test critical database imports work."""
        try:
            import asyncpg
            import psycopg2
            from sqlalchemy import MetaData, create_engine, text
            from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
            from sqlalchemy.orm import Session, sessionmaker

            # These must all succeed
            assert all(
                [
                    create_engine,
                    text,
                    MetaData,
                    sessionmaker,
                    Session,
                    create_async_engine,
                    AsyncSession,
                    asyncpg,
                    psycopg2,
                ]
            )
        except ImportError as e:
            pytest.fail(f"Critical database import failed: {e}")
        except Exception as e:
            pytest.fail(f"Critical database import setup failed: {e}")

    def test_critical_auth_imports(self):
        """Test critical authentication imports work."""
        try:
            import jwt as pyjwt
            from jose import JWTError, jwt
            from passlib.context import CryptContext

            # These must all succeed
            assert all([CryptContext, jwt, JWTError, pyjwt])
        except ImportError as e:
            pytest.fail(f"Critical auth import failed: {e}")


if __name__ == "__main__":
    # Run tests to verify dependencies
    # This should fail initially if dependencies are missing
    pytest.main([__file__, "-v", "--tb=short"])
